#!/usr/bin/env ruby

require 'rubygems'
require 'optparse'
require 'fileutils'

DumpDataFile = 'dump.js'

$options = {}
$dump_options = {}
$script_path = File.expand_path(File.dirname(__FILE__))
$dump_script = File.join $script_path, 'dump_to_json.rb'
if !File.exists? $dump_script
    STDERR.puts "FATAL: could not find script #{$dump_script}"
    exit 1
end
$resource_path = File.join(File.dirname($script_path), 'resources')
if !File.exists? $resource_path
    STDERR.puts "FATAL: could not find resources at: #{$resource_path}"
    exit 1
end
$templates = {}
%w(html js css).each{|ext|
    template = File.join($resource_path, "dump_viewer/dump_viewer.#{ext}")
    if !File.exists?(template)
        STDERR.puts "FATAL: could not find template #{template}"
        exit 1
    end
    $templates[:"#{ext}"] = template
}

optparse = OptionParser.new do |opts|

    opts.banner = "#{$0} [OPTIONS] DUMP_FILE"

    opts.on '', '--max-row ROW', 'Max Neuron Row' do |max_row|
        max_row = max_row.to_i
        if max_row < 1
            STDERR.puts "--max-row must be >= 1"
            exit 1
        end
        $dump_options[:max_row] = max_row
    end

    opts.on '', '--include-last-rows',
            'If --max-row is used,',
            'include last rows (output_height - max_row)' do
        $dump_options[:include_last_rows] = true
    end

    opts.on '', '--max-feature FEAT', 'Max Layer Feature' do |max_feat|
        max_feat = max_feat.to_i
        if max_feat < 1
            STDERR.puts "--max-feature must be >= 1"
            exit 1
        end
        $dump_options[:max_feat] = max_feat
    end

    opts.on '', '--coord-order ORDER',
            "Coordinate order (ex. 'x,y' or 'auto')",
            "Default: #{$options[:coord_order]}" do |order|
        order = order.downcase
        if order != 'auto'
            coords = order.split ','
            coords &= %w(x y)
            if coords.length != 2
                STDERR.puts "Invalid --coord-order"
                exit 1
            end
        end
        $dump_options[:coord_order] = order
    end

    opts.on '-h', '--help', 'Print this help' do
        STDERR.puts opts
        exit 1
    end

end

optparse.parse!

$tmpdir = "/tmp/psyc_dump_#{Time.now.to_i}"
FileUtils.mkdir_p $tmpdir if !File.exists? $tmpdir
$js_output = File.join($tmpdir, DumpDataFile)

$dump_file = ARGV.first
if !$dump_file || $dump_file.strip.empty?
    STDERR.puts "Missing DUMP_FILE"
    exit 1
end

if !File.exists? $dump_file
    STDERR.puts "File #{$dump_file} not found!"
    exit 1
end

cmd = [
    $dump_script, "--js DumpData", "-o #{$js_output}"
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
if !$?.success?
    STDERR.puts "Failed to generate #{DumpDataFile}"
    exit 1
end
html = File.read $templates[:html]
html.gsub! '$DUMP_JS_PATH', DumpDataFile
html_dest = File.join  $tmpdir, 'html_viewer.html'
puts "Generating html_viewer.html"
File.open(html_dest, 'w:utf-8'){|f| f.write(html)}
[:js, :css].each{|type|
    template = $templates[type]
    fname = File.basename template
    dest = File.join $tmpdir, fname
    puts "Generating #{fname}"
    FileUtils.cp template, dest
}
puts "Psyc Dump Viewer generated at: #{html_dest}"
