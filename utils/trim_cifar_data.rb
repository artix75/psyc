#!/usr/bin/env ruby

IMAGE_SIZE = 32 * 32 * 3
ELEM_SIZE = IMAGE_SIZE + 1
$script_path = File.expand_path(File.dirname(__FILE__))

def print_help
    STDERR.puts "Usage: #{$0} DATA_FILE NUM_ELEMENTS [OUTPUT]"
end

if ARGV.include?('-h') || ARGV.include?('--help')
    print_help
    exit 1
end

$data_file, $num_elems, $output = ARGV
if !$data_file || !$num_elems
    print_help
    exit 1
end
if !File.exists? $data_file
    STDERR.puts "Could not find #{$data_file}"
    exit 1
end
$num_elems = $num_elems.to_i
if $num_elems <= 0
    STDERR.puts "NUM_ELEMENTS must be >= 1"
    exit 1
end
if !$output
    fname = File.basename $data_file
    ext = File.extname fname
    fname = fname.sub /#{Regexp.escape(ext)}$/, ".0-#{$num_elems}#{ext}"
    $output = File.join '/tmp', fname
end
numbytes = $num_elems * ELEM_SIZE
bytes = File.read $data_file, numbytes
if bytes.encoding.to_s.downcase != 'ascii-8bit'
    "FATAL: read bytes encoding is not 'ascii-8bit' (got #{bytes.encoding})"
    exit 1
end
if bytes.length != numbytes
    "FATAL: read bytes length should be #{numbytes}, got #{bytes.length}"
    exit 1
end

File.open($output, 'w:ascii-8bit'){|f|
    f.write bytes
}

if !File.exists? $output
    STDERR.puts "Failed to generate output!"
    exit 1
end

if File.size($output) != numbytes
    err = "FATAL: output size should be #{numbytes}, got " +
          "#{File.size($output)}, in\n"+"       #{$output}"
    STDERR.puts err
    exit 1
end

puts "Trimmed data written to: #{$output}"
