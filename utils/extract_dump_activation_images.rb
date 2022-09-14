#!/usr/bin/env ruby

require 'rubygems'
require 'optparse'
require 'fileutils'
require 'json'

require 'rmagick'

$options = {}
$dump_options = {}
$script_path = File.expand_path(File.dirname(__FILE__))
$dump_script = File.join $script_path, 'dump_to_json.rb'
if !File.exists? $dump_script
    STDERR.puts "FATAL: could not find script #{$dump_script}"
    exit 1
end

optparse = OptionParser.new do |opts|

    opts.banner = "#{$0} [OPTIONS] DUMP_FILE"

    opts.on '-h', '--help', 'Print this help' do
        STDERR.puts opts
        exit 1
    end

end

optparse.parse!

$dump_file = ARGV[0]
if !$dump_file
    STDERR.puts optparse
    exit 1
end
if !File.exists? $dump_file
    STDERR.puts "File #{$dump_file} not found!"
    exit 1
end

dump_dir = File.expand_path(File.dirname($dump_file))
ext = File.extname $dump_file
dirname = File.basename $dump_file
fstem = dirname
if ext && !ext.empty?
    dirname = dirname.sub /#{Regexp.escape(ext)}$/, ''
    fstem = dirname.dup
end
dirname = "#{dirname}_images"
$tmpdir = File.join dump_dir, dirname
if !File.exists? $tmpdir
    FileUtils.mkdir_p $tmpdir
end
$json_output = File.join($tmpdir, 'dump.json')

cmd = [
    $dump_script, "-o #{$json_output}"
] + $dump_options.map{|opt, val|
    opt = opt.to_s.split(/_+/).join('-')
    if opt.length == 1
        opt = "-#{opt}"
    else
        opt = "--#{opt}"
    end
    if val == true
        opt
    else
        "#{opt} #{val}"
    end
}
cmd << $dump_file.inspect
cmd = cmd.join(' ')
puts cmd
system cmd
if !$?.success? || !File.exists?($json_output)
    STDERR.puts "Failed to generate #{$json_output}"
    exit 1
end
$quantum_range = Magick::QuantumRange.to_f
$dump = JSON.parse(File.read($json_output), symbolize_names: true)
$layers = $dump[:layer]
$layers.each_with_index{|l, i|
    size = l[:size]
    index = l[:index]
    features = l[:features] || 1
    features = 1 if features < 1
    feat_size = size / features
    type = l[:type]
    is_rgb = (index == 0 || i == 0) && features == 3
    activations = l[:activations]
    puts "Layer[#{index || i}] #{type}, size: #{size}, features: #{features}, "+
         "feature_size: #{feat_size}, rgb: #{is_rgb}, " +
         "activations: #{activations.length}"
    output_size = l[:output_size]
    if !output_size && (next_l = $layers[i + 1])
        output_size = next_l[:input_size]
    end
    if !output_size
        puts " - Layer cannot be presented as image"
        next
    end
    output_size = output_size.split('x').map{|v| v.to_i}
    imgdata = []
    images = []
    err = nil
    max_activation = activations.max
    activations.each_with_index{|a, i|
        feat_idx = i / feat_size
        nidx = i % feat_size
        if max_activation > 1
            a = (a.to_f / max_activation)
        end
        val = (a * $quantum_range)
        if val < 0
            err = "Negative value #{val} (activation: #{a}) for layer " +
                  "#{index}, neuron #{i}"
            break
        end
        if is_rgb
            channel = (imgdata[nidx] ||= [0, 0, 0])
            channel[feat_idx] = val
        else
            if nidx == 0 && feat_idx > 0
                images << imgdata
                imgdata = []
            end
            if nidx < feat_size
                imgdata << val
            end
        end
    }
    if err
        STDERR.puts err
        next
    end
    if imgdata.length > 0 && images.length < features
        images << imgdata
    end
    if images.length < 1
        puts " -> No images extracted!"
        next
    end
    fidx = 0
    images = images.map{|imgdata|
        image = Magick::Image.new *output_size
        pixels = imgdata.map{|val|
            if val.is_a? Array
                Magick::Pixel.new *val
            else
                Magick::Pixel.new val, val, val
            end
        }
        image.store_pixels(0, 0, output_size[0], output_size[1], pixels)
        imgname = "#{fstem}-layer-#{index}-f#{fidx}.jpg"
        fidx += 1
        dest = File.join($tmpdir, imgname)
        puts " -> Writing image: #{dest}"
        image.write "jpg:#{dest}"
        image
    }
}
