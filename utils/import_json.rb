#!/usr/bin/env ruby

require 'rubygems'
require 'json'

DOUBLE_FORMAT = '%.15e'

json_file = ARGV[0]
if !json_file
    puts "Usage: #{$0} JSON_FILE [OUTPUT_FILE]"
    exit 1
end
json_file = File.expand_path json_file
net = JSON.parse(File.read(json_file), symbolize_names: true)
if !net.is_a? Hash
    puts "Invalid file!"
    exit 1
end
layout = net[:layout]
if !layout
    puts "Missing layout!"
    exit 1
end
layers = net[:layers]
if !layers
    puts "Missing :layers!"
    exit 1
end
out = "#{layout.length}:" + layout.join(",") + "\n"
layers[1..-1].each_with_index{|l, i|
    #size = l[:size]
    neurons = l[:neurons]
    lstr = neurons.map{|n|
        bias = DOUBLE_FORMAT % n[:bias]
        weights = n[:weights].map{|w| DOUBLE_FORMAT % w}
        "#{bias}|#{weights.join(',')}"
    }.join("\n")
    out << lstr
}
out_file = ARGV[1] || "./network.#{Time.now.to_i}.data"
File.open(out_file, 'w'){|f|
    f.write out
}
puts "File written to #{out_file}"

