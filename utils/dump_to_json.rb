#!/usr/bin/env ruby

require 'rubygems'
require 'optparse'
require 'json'

DefaultCoordOrder = 'x,y'
$options = {coord_order: 'auto'}

optparse = OptionParser.new do |opts|

    opts.banner = "#{$0} [OPTIONS] DUMP_FILE"

    opts.on '-o', '--output FILE', 'Output File' do |opath|
        $options[:output] = opath
    end

    opts.on '', '--js JS_VAR',
            'Output a Javascript file instead of a JSON and',
            'puts everything into the variable JS_VAR' do |js_var|
        js_var = js_var.strip
        if js_var.empty?
            STDERR.puts "Invalid JS_VAR (empty)"
            exit 1
        end
        if js_var[/\s/]
            STDERR.puts "Invalid JS_VAR: spaces not allowed"
            exit 1
        end
        $options[:js_var] = js_var
    end

    opts.on '', '--max-row ROW', 'Max Neuron Row' do |max_row|
        max_row = max_row.to_i
        if max_row < 1
            STDERR.puts "--max-row must be >= 1"
            exit 1
        end
        $options[:max_row] = max_row
    end

    opts.on '', '--include-last-rows',
            'If --max-row is used,',
            'include last rows (output_height - max_row)' do
        $options[:include_last_rows] = true
    end

    opts.on '', '--max-feature FEAT', 'Max Layer Feature' do |max_feat|
        max_feat = max_feat.to_i
        if max_feat < 1
            STDERR.puts "--max-feature must be >= 1"
            exit 1
        end
        $options[:max_feat] = max_feat
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
        $options[:coord_order] = order
    end

    opts.on '', '--pretty', 'Pretty readable output' do
        $options[:pretty] = true
    end

    opts.on '-h', '--help', 'Print this help' do
        STDERR.puts opts
        exit 1
    end

end

optparse.parse!

$dump_file = ARGV.first
if !$dump_file || $dump_file.strip.empty?
    STDERR.puts "Missing DUMP_FILE"
    exit 1
end

if !File.exists? $dump_file
    STDERR.puts "File #{$dump_file} not found!"
    exit 1
end

def parse_value(val)
    val.strip!
    if val[/^[\-\+]?\d+$/]
        val.to_i
    elsif val[/^[\-\+]?\d+\.\d+$/]
        val.to_f
    elsif val[/^[\-\+]?\d+\.\d+[eE][\-\+]\d+/]
        val.to_f
    elsif val == 'true'
        true
    elsif val == false
        false
    elsif val == 'null'
        nil
    elsif val.empty?
        nil
    else
        val
    end
end

puts "Loading dump file..."
$dump = File.read $dump_file
$data = {}
ScalarValueExp=/^([a-zA-Z_]+)=([^,\(\)]+),?/
VectorValueExp=/^([a-zA-Z_]+)=\(([^\(\)]*)\),?/
EmptyValueExp1=/^([a-zA-Z_]+)=,/
EmptyValueExp2=/^([a-zA-Z_]+)=$/
puts "Splitting lines..."
$lines = $dump.strip.split(/\n+/)
totlines = $lines.length
puts "Dump file has #{totlines} line(s)"
puts "Parsing dump file..."
max_row = $options[:max_row]
include_last_rows = ($options[:include_last_rows] == true)
max_feat = $options[:max_feat]
coord_order = $options[:coord_order]
auto_coord_order = (coord_order == 'auto')
if !auto_coord_order
    coord_order = coord_order.split(',')
else
    coord_order = nil
end
$step_map = []
last_step_phase = nil
last_step_func = nil
last_step_layer = nil
$lines.each_with_index{|l, idx|
    lineno = idx + 1
    perc = ((lineno / totlines.to_f) * 100).round
    STDOUT.flush
    print "\rLine #{lineno}/#{totlines} (#{perc}%)"
    STDOUT.flush
    l = l.strip
    if l[0, 1] == '#'
        if auto_coord_order &&
           (match = l.match(/coordinate_order\s*=\s*([xy],[xy])/)) &&
           match[1]
            coord_order = match[1].split ','
        end
        next
    end
    label,row_data = l.split ':', 2
    next if !label || label.strip.empty?
    row_data ||= ''
    label = :"#{label}"
    data = {}
    max_cycles = 100
    cycle = 0
    while row_data.length > 0
        if cycle >= max_cycles
            STDERR.puts "FATAL: stuck while  parse line #{lineno}: #{l}"
            STDERR.puts "   row_data: #{row_data}"
            STDERR.flush
            STDOUT.flush
            puts "\n"
            exit 1
        end
        if (match = row_data.match(ScalarValueExp))
            row_data.sub! match[0], ''
            var, val = match[1], match[2]
            next if var.strip.empty?
            val = parse_value val
            data[:"#{var}"] = val
        elsif (match = row_data.match(VectorValueExp))
            row_data.sub! match[0], ''
            var, val = match[1], match[2]
            val = val.split(',').map{|v| parse_value(v)}
            data[:"#{var}"] = val
        elsif (match = row_data.match(EmptyValueExp1)) ||
              (match = row_data.match(EmptyValueExp2))
            row_data.sub! match[0], ''
            var = match[1]
            data[:"#{var}"] = nil
        else
            puts "\n"
            STDOUT.flush
            STDERR.puts "ERROR: Invalid line at #{lineno}:\n#{l}"
            exit 1
        end
        cycle += 1
    end
    data[:lineno] = lineno
    neuron = data[:neuron]
    layer = nil
    layer_idx = nil
    feat_idx = nil
    nidx = nil
    input_sz = nil
    output_sz = nil
    if neuron
        $neuron_comps ||= {}
        ncomps = $neuron_comps[neuron]
        if !ncomps
            ncomps = $neuron_comps[neuron] = neuron.split '-'
        end
        if ncomps.length == 3
            layer_idx, feat_idx, nidx = ncomps.map{|n| n.to_i}
        elsif ncomps.length == 2
            layer_idx, nidx = ncomps.map{|n| n.to_i}
        end
        if layer_idx && ($layers = $data[:layer])
            layer = $layers[layer_idx]
            if layer
                if (isz = layer[:input_size])
                    input_sz = isz.split('x').map{|n| n.to_i}
                end
                if (osz = layer[:output_size])
                    output_sz = osz.split('x').map{|n| n.to_i}
                end
            end
        end
    end
    if max_feat && feat_idx
        next if feat_idx >= max_feat
    end
    if max_row && (pos = (data[:pos] || data[:neuron_pos]))
        order = coord_order || DefaultCoordOrder
        y_idx = order.index('y')
        if (y = pos[y_idx])
            if include_last_rows && output_sz && (oh = output_sz[y_idx])
                min_row = oh - max_row
                next if y >= max_row && y < min_row
            else
                next if y >= max_row
            end
        end
    end
    if (current = $data[label])
        if !current.is_a? Array
            $data[label] = [current]
        end
        $data[label] << data
    else
        $data[label] = data
    end
    if label == :step && ($steps = $data[:step])
        $steps = [$steps] if !$steps.is_a?(Array)
        step_phase = data[:phase]
        step_func = data[:func]
        step_layer = data[:layer]
        if step_phase != last_step_phase ||
           step_func != last_step_func ||
           step_layer != last_step_layer
            step_idx = $steps.length - 1
            map_point = {
                step_index: step_idx,
                phase: step_phase,
                func: step_func,
                layer: step_layer,
            }
            $step_map << map_point
        end
        last_step_phase = step_phase
        last_step_func = step_func
        last_step_layer = step_layer
    end
}
puts "\nDone!"
STDOUT.flush
$options[:coord_order] = coord_order if coord_order
$data[:script_options] = $options
$data[:step_map] = $step_map

$json = nil
if $options[:pretty]
    $json = JSON.pretty_generate $data
else
    $json = $data.to_json
end
outfile = $options[:output]
output_content = $json
js_var = $options[:js_var]
if js_var
    output_content = "var #{js_var} = #{$json};"
end
if !outfile || outfile.strip.empty?
    dirname = File.dirname $dump_file
    dirname = File.expand_path dirname
    fname = File.basename $dump_file
    ext = File.extname fname
    if ext && !ext.strip.empty?
        fstem = fname.sub /#{Regexp.escape(ext)}$/, ''
    else
        fstem = fname.dup
    end
    outfname = nil
    if js_var
        outfname = "#{fstem}.js"
    else
        outfname = "#{fstem}.json"
    end
    outfile = File.join dirname, outfname
end
File.open(outfile, 'w:utf-8'){|f| f.write(output_content)}
puts "Written to: #{outfile.inspect}"
