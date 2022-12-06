#!/usr/bin/env ruby
require 'csv'
require 'optparse'
require 'json'

VOCABULARY_SIZE = 5000
START_TOKEN = '__S_START__'
END_TOKEN = '__S_END__'
UNKNOWN_TOKEN = '__UNK__'
MAX_SENTENCES = 0#nil
TEST_DATALEN_PERC = 0.10
EVAL_DATALEN_PERC = 0.10

def print_progress(msg, idx, total)
    msg = "#{msg} #{idx + 1}/#{total}"
    l = msg.length
    spaces = ' ' * (39 - l)
    print "\r#{msg}#{spaces}"
    if idx == (total - 1)
        puts ""
    end
end

def tokenize_sentence(s)
    s = s.gsub /[:;\.,\(\)\{\}\[\]]/, ''
    #s = s.gsub /[\P{L}]+/u, ''
    s = s.gsub /(['\-\+\*\/\|])/, ' \1 '
    s.split /\s+/
end

def sentence2vec(s, as_float: false, char_mode: false)
    unkn_tok = !char_mode ? UNKNOWN_TOKEN : "\n"
    unkn_idx = $word_index[unkn_tok]
    s = tokenize_sentences(s) if s.is_a?(String) && !char_mode
    s.map{|w|
        i = $word_index[w] || unkn_idx
        (as_float ? i.to_f : i)
    }
end

def vec2sentence(vec)
    if vec[0].is_a? Array
        vec = vec.map{|v| max_idx(v)}
    end
    vec.map{|idx|
        indexed_words[idx] || UNKNOWN_TOKEN
    }.join("\n")
end

def one_hot(array)
    array.map{|idx|
        vec = Array.new VOCABULARY_SIZE, 0
        vec[idx] = 1
        vec
    }
end

def max_idx(vec)
    vec.index(vec.max)
end

def dump_sentence_vec(vec, slen, ylen, sidx: nil, fixed_ylen: nil, name: nil)
    prfx = (name ? "#{name} " : '')
    print "#{prfx}Sentence"
    if sidx
        print " #{sidx}"
    end
    print ":\n"
    if vec.length != (slen + ylen)
        STDERR.puts "Error (dump_sentence_vec): Invalid sentence length. " +
                    "Vector length #{vec.length} != #{slen + ylen} " +
                    "(#{slen} + #{ylen})" + (name ? " in #{name} data" : '')
        return
    end
    xvec = vec[0, slen]
    yvec = vec[slen..-1]
    xwords = xvec.map{|widx|
        $indexed_words[widx].inspect
    }
    ywords = yvec.map{|widx|
        $indexed_words[widx].inspect
    }
    puts "#{prfx}Feed sequence (X)"
    puts "  Data:  [#{xvec.join(', ')}]"
    puts "  Words: [#{xwords.join(', ')}]"
    puts "#{prfx}Label sequence (Y)"
    puts "  Data:  [#{yvec.join(', ')}]"
    if !fixed_ylen
        puts "  Words: [#{ywords.join(', ')}]"
    end
end

def do_verify(c, dump_sentences: false)
    vars = {}
    var_names = %w(
        VOCABULARY_SIZE
        SENTENCE_COUNT
        TRAIN_DATA_LEN
        EVAL_DATA_LEN
        TEST_DATA_LEN
        TRAIN_SENTENCES
        EVAL_SENTENCES
        TEST_SENTENCES
    )
    var_names.each{|var|
        m = c.match(/#define\s+#{var}\s+(\d+)/)
        if !m
            STDERR.puts "Verify: Couldn't find #{var}, file is not valid"
            return false
        end
        vars[var] = m[1].to_i
    }
    # If label (y) sequence length is the same of inputs (x), fixed_ylen is nil
    # or zero.
    fixed_ylen = nil
    if (m = c.match(/#define\s+Y_SEQUENCE_COUNT\s+(\d+)/))
        fixed_ylen = m[1].to_i
        fixed_ylen = nil if fixed_ylen <= 0
    end
    if dump_sentences && !$indexed_words
        wrd_match = c.match(
            /char\s*\*?\s*#{$options[:words_var]}\[\]\s*=\s*\{([^\{\}]+)\};/
        )
        if !wrd_match
            STDERR.puts "Verify: Couldn't find #{$options[:words_var]}, " +
                        "cannot dump sentences"
            dump_sentences = false
        end
        matched_words = wrd_match[1]
        if $options[:mode] == :characters
            matched_words = matched_words.gsub "'", '"'
        end
        begin
            words_json = '[' + matched_words + ']'
            $indexed_words = JSON.parse words_json
        rescue Exception => e
            STDERR.puts "ERROR: Failed to read words, cannot dump sentences"
            STDERR.puts e.to_s
            dump_sentences = false
        end
    end
    #words = c.match(/char \* #{$options[:words_var]}\[\] = \{(.*+?)\};/
    tr_data = c.match(
        /PSFloat #{$options[:train_data_var]}\[\] = \{([\d\.,\s]+)\};/
    )
    if !tr_data
        STDERR.puts "Verify: Couldn't find #{$options[:train_data_var]}, "+
                    "file is not valid"
        return false
    end
    tr_data = tr_data[1].split(',').map{|n| n.strip.to_f}
    ev_data = c.match(
        /PSFloat #{$options[:eval_data_var]}\[\] = \{([\d\.,\s]+)\};/
    )
    if !ev_data
        STDERR.puts "Verify: Couldn't find #{$options[:eval_data_var]}, "+
                    "file is not valid"
        return false
    end
    ev_data = ev_data[1].split(',').map{|n| n.strip.to_f}
    ts_data = c.match(
        /PSFloat #{$options[:test_data_var]}\[\] = \{([\d\.,\s]+)\};/
    )
    if !ts_data
        STDERR.puts "Verify: Couldn't find #{$options[:test_data_var]}, "+
                    "file is not valid"
        return false
    end
    ts_data = ts_data[1].split(',').map{|n| n.strip.to_f}
    if tr_data.length != vars['TRAIN_DATA_LEN']
        STDERR.puts "Verify: #{tr_data.length} != TRAIN_DATA_LEN"
        return false
    end
    tr_data_len = tr_data.shift.to_i
    if tr_data_len != vars['TRAIN_SENTENCES']
        STDERR.puts "Verify: #{tr_data_len} != TRAIN_SENTENCES"
        return false
    end
    train_label_stats = {}
    eval_label_stats = {}
    test_label_stats = {}
    train_sequences = 0
    eval_sequences = 0
    test_sequences = 0
    counted = 1
    sidx = 0
    while s_len = tr_data.shift
        s_len = s_len.to_i
        y_len = fixed_ylen || s_len
        seqlen = s_len + y_len
        if dump_sentences
            dump_sentence_vec tr_data[0, seqlen], s_len, y_len, sidx: sidx,
                              fixed_ylen: fixed_ylen, name: 'Train'
        end
        yvec = tr_data[s_len, y_len]
        lbl = yvec.length > 1 ? yvec : yvec[0]
        train_label_stats[lbl] ||= 0
        train_label_stats[lbl] += 1
        train_sequences += 1
        counted += 1
        len = tr_data.length
        tr_data = tr_data[seqlen..-1] || []
        sidx += 1
        counted += (len - tr_data.length)
    end
    if counted != vars['TRAIN_DATA_LEN']
        STDERR.puts "Verify: #{$options[:train_data_var]} is invalid"
        return false
    end
    if train_sequences != vars['TRAIN_SENTENCES']
        STDERR.puts "Verify: counted train sequences != TRAIN_SENTENCES " +
                    "(#{train_sequences} != #{vars['TRAIN_SENTENCES']})"
        return false
    end

    if ev_data.length != vars['EVAL_DATA_LEN']
        STDERR.puts "Verify: #{ev_data.length} != EVAL_DATA_LEN"
        return false
    end
    ev_data_len = ev_data.shift.to_i
    if ev_data_len != vars['EVAL_SENTENCES']
        STDERR.puts "Verify: #{ev_data_len} != EVAL_SENTENCES"
        return false
    end
    counted = 1
    sidx = 0
    while s_len = ev_data.shift
        s_len = s_len.to_i
        y_len = fixed_ylen || s_len
        seqlen = s_len + y_len
        if dump_sentences
            dump_sentence_vec ev_data[0, seqlen], s_len, y_len, sidx: sidx,
                              fixed_ylen: fixed_ylen, name: 'Eval'
        end
        yvec = ev_data[s_len, y_len]
        lbl = yvec.length > 1 ? yvec : yvec[0]
        eval_label_stats[lbl] ||= 0
        eval_label_stats[lbl] += 1
        eval_sequences += 1
        counted += 1
        len = ev_data.length
        ev_data = ev_data[seqlen..-1] || []
        sidx += 1
        counted += (len - ev_data.length)
    end
    if counted != vars['EVAL_DATA_LEN']
        STDERR.puts "Verify: #{$options[:eval_data_var]} is invalid"
        return false
    end
    if eval_sequences != vars['EVAL_SENTENCES']
        STDERR.puts "Verify: counted eval sequences != EVAL_SENTENCES " +
                    "(#{eval_sequences} != #{vars['EVAL_SENTENCES']})"
        return false
    end

    if ts_data.length != vars['TEST_DATA_LEN']
        STDERR.puts "Verify: #{ts_data.length} != TEST_DATA_LEN"
        return false
    end
    ts_data_len = ts_data.shift.to_i
    if ts_data_len != vars['TEST_SENTENCES']
        STDERR.puts "Verify: #{ts_data_len} != TEST_SENTENCES"
        return false
    end
    counted = 1
    sidx = 0
    while s_len = ts_data.shift
        s_len = s_len.to_i
        y_len = fixed_ylen || s_len
        seqlen = s_len + y_len
        if dump_sentences
            dump_sentence_vec ts_data[0, seqlen], s_len, y_len, sidx: sidx,
                              fixed_ylen: fixed_ylen, name: 'Test'
        end
        yvec = ts_data[s_len, y_len]
        lbl = yvec.length > 1 ? yvec : yvec[0]
        test_label_stats[lbl] ||= 0
        test_label_stats[lbl] += 1
        test_sequences += 1
        counted += 1
        len = ts_data.length
        ts_data = ts_data[seqlen..-1] || []
        sidx += 1
        counted += (len - ts_data.length)
    end
    if counted != vars['TEST_DATA_LEN']
        STDERR.puts "Verify: #{$options[:test_data_var]} is invalid"
        return false
    end
    if test_sequences != vars['TEST_SENTENCES']
        STDERR.puts "Verify: counted tes sequences != TEST_SENTENCES " +
                    "(#{test_sequences} != #{vars['TEST_SENTENCES']})"
        return false
    end

    puts "Training Labels:"
    train_label_stats.each{|lbl, lcount|
        perc = ((lcount.to_f / train_sequences) * 100).round
        lbl = lbl.inspect
        if lbl.length > 50
            lbl = lbl[0, 47] + '...'
        end
        puts " - #{lcount} (#{perc}%) - #{lbl}"
    }
    puts "Evaluation Labels:"
    eval_label_stats.each{|lbl, lcount|
        perc = ((lcount.to_f / eval_sequences) * 100).round
        lbl = lbl.inspect
        if lbl.length > 50
            lbl = lbl[0, 47] + '...'
        end
        puts " - #{lcount} (#{perc}%) - #{lbl}"
    }
    puts "Testing Labels:"
    test_label_stats.each{|lbl, lcount|
        perc = ((lcount.to_f / test_sequences) * 100).round
        lbl = lbl.inspect
        if lbl.length > 50
            lbl = lbl[0, 47] + '...'
        end
        puts " - #{lcount} (#{perc}%) - #{lbl}"
    }
    puts "File is valid!"
    true
end

def load_data_from(filename, opts = {})
    label = opts[:label]
    lbldescr = ''
    lbl_offset = nil
    if label
        if label.is_a?(Array)
            lbldescr = " (label: #{label.join(',')[0,10].strip})"
        else
            lbldescr = " (label: #{label.to_s})"
        end
        $training_label_offsets ||= {}
        $training_label_offsets[label] ||= []
        lbl_offset = [$training_data ? $training_data.length : 0]
    end
    puts "Reading #{filename}#{lbldescr}"
    data = File.read(filename)
    whole_file = opts[:whole_file]
    char_mode = opts[:mode] == :characters
    start_end_tokens = opts[:start_end_tokens] != false
    if filename[/\.csv$/i]
        data = CSV.parse(File.read(filename))
        data.shift
        data = data.join(' ') if whole_file
    else
        if !whole_file
            sep = opts[:separator] || /\n+/
            data = data.split sep
        else
            data = [data]
        end
    end
    $sentences ||= []
    $tokenized_sentences ||= []
    $words ||= {}
    sent_len = $sentences.length
    puts "Parsing File #{filename}..."
    data_len = data.length
    max_s = opts[:max_sentences] || $options[:max_sentences]
    data.each_with_index{|txt, idx|
        txt = txt[0] if txt.is_a? Array
        next if !txt || txt.strip.empty?
        #puts txt
        txt = txt.downcase
        if !whole_file && !char_mode
            processed = txt.gsub /([a-zA-Z])(\s*\.\s*)([a-zA-Z])/, '\1__DOT__\3'
            #puts processed
            #puts "SENTENCES:"
            sentences = processed.split '__DOT__'
        else
            processed = txt.gsub /([a-zA-Z])(\s*\.\s*)([a-zA-Z])/, '\1 . \3'
            sentences = [processed]
        end
        sentences.each{|sent|
            #puts sent
            #puts '-'*30
            if !char_mode && start_end_tokens != false
                $sentences << "#{START_TOKEN} #{sent} #{END_TOKEN}"
            else
                $sentences << sent
            end
        }
        if max_s > 0 && $sentences.length > max_s
            $sentences = $sentences[0, max_s]
            break
        end
        print_progress "Row", idx, data_len
    #    break if $sentences.length > 10
    }
    #p $sentences
    puts "\nTokenizing sentences..."
    _idx = 0
    _tot = $sentences.length
    tokenized = []
    $sentences[sent_len..-1].each_with_index{|s, _idx|
        print_progress "Sentence", _idx, _tot
        if !char_mode
            s = tokenize_sentence s
        else
            s = s.chars
        end
        next if s.length == 0
        tokenized << s
    }
    puts ''
    #p $tokenized_sentences
    #$words = {}
    tokenized.flatten.each{|w|
        $words[w] ||= 0
        $words[w] += 1
    }
    $words["\n"] = 1 if char_mode

    $tokenized_sentences += tokenized

    puts "Creating Vocabulary..."

    vocabulary_size = $options[:vocabulary_size]
    $vocabulary = $words.to_a.sort_by{|v| v[1]}.reverse[0, vocabulary_size - 1]
    $vocabulary << UNKNOWN_TOKEN if !char_mode 
    $word_index = {}
    $indexed_words = []
    $vocabulary.each_with_index{|v, i|
        word, freq = v
        $word_index[word] = i
        $indexed_words << word
    }
    puts "Remapping sentences"
    _idx = 0
    _len = $tokenized_sentences.length
    $tokenized_sentences.map!{|sent|
        print_progress "Sentence", _idx, _tot
        _idx += 1
        sent.map{|w|
            if $word_index[w]
                w
            else
                char_mode ? "\n" : UNKNOWN_TOKEN
            end
        }
    }
    puts ''
    #p $tokenized_sentences[0,5]
    _idx = 0
    _len = $tokenized_sentences.length
    puts "\nCreating training data..."
    $training_data_indexed ||= []
    $training_data ||= []
    $training_data += tokenized.map{|sent|
        print_progress "Sentence", _idx, _tot
        _idx += 1
        sent = sentence2vec sent, as_float: true, char_mode: char_mode
        x = sent[0..-2]
        y = sent[1..-1]
        if label
            if !label.is_a? Array
                #y = Array.new y.length, label.to_f
                x = sent[0..-1]
                y = [label.to_f]
            else
                lengths = [y.length, label.length].sort
                y = (label * (lengths[1] / lengths[0]))[0, y.length]
            end
            $labels_stats ||= {}
            $labels_stats[label] ||= 0
            $labels_stats[label] += 1
        end
        pair = [x, y]
        $training_data_indexed << pair
        pair
    }
    if lbl_offset && $training_data.length > 0
        lbl_offset << ($training_data.length - 1)
        $training_label_offsets[label] << lbl_offset
    end
    puts ''
end

$options = {
    vocabulary_size: VOCABULARY_SIZE,
    max_sentences: MAX_SENTENCES,
    words_var: 'words',
    train_data_var: 'training_data',
    eval_data_var: 'validation_data',
    test_data_var: 'test_data',
    mode: :words
}

optparse = OptionParser.new do |opts|

    opts.banner = "Usage: #{File.basename($0)} [options]"

    opts.on '-o', '--output OUTPUT', 'Output File' do |out|
        $options[:output] = out
    end

    opts.on '-s', '--vocabulary-size SIZE',
            "Vocabulary size (def. #{VOCABULARY_SIZE})" do |size|
        size = size.to_i
        if size.zero?
            STDERR.puts "Invalid size"
            exit 1
        end
        $options[:vocabulary_size] = size
    end

    opts.on '', '--max-sentences MAX',
            "Maximum sentences, 0 for no limit (def. #{MAX_SENTENCES})" do |max|
        max = max.to_i
        if max.zero?
            STDERR.puts "Invalid max sentences"
            exit 1
        end
        $options[:max_sentences] = max
    end

    opts.on '', '--words-var VAR',
            "Words C variable name (def. #{$options[:words_var]})" do |var|
        if !var[/^[_a-zA-Z][_a-zA-Z0-9]*$/]
            STDERR.puts "Invalid word varabile name"
            exit 1
        end
        $options[:words_var] = var
    end

    opts.on '', '--data-var VAR',
            "Training data C var. (def. #{$options[:train_data_var]})" do |var|
        if !var[/^[_a-zA-Z][_a-zA-Z0-9]*$/]
            STDERR.puts "Invalid training data varabile name"
            exit 1
        end
        $options[:train_data_var] = var
    end

    opts.on '', '--validation-data-var VAR',
            "Validation data C var. (def. #{$options[:eval_data_var]})" do |var|
        if !var[/^[_a-zA-Z][_a-zA-Z0-9]*$/]
            STDERR.puts "Invalid validation data varabile name"
            exit 1
        end
        $options[:eval_data_var] = var
    end

    opts.on '', '--test-data-var VAR',
            "Test data C var. (def. #{$options[:test_data_var]})" do |var|
        if !var[/^[_a-zA-Z][_a-zA-Z0-9]*$/]
            STDERR.puts "Invalid test data varabile name"
            exit 1
        end
        $options[:test_data_var] = var
    end

    opts.on '', '--same-dataset',
            'Train also with test/validation datasets' do |same_dataset|
        $options[:same_dataset] = true
    end

    opts.on '', '--whole-file', 'Import whole file(s)' do
        $options[:whole_file] = true
    end

    opts.on '-l', '--labels LABELS', 'Add custom labels' do |labels|
        $options[:labels] ||= []
        $options[:labels] << labels
    end

    opts.on '', '--shuffle', 'Shuffle Data' do
        $options[:shuffle] = true
    end

    opts.on '-m', '--mode MODE', 'Mode: words|characters, def. words' do |mode|
        if !%w(words characters).include? mode
            STDERR.puts "Invalid mode: #{mode}"
            exit 1
        end
        $options[:mode] = :"#{mode}"
    end

    opts.on '', '--no-start-end-tokens',
            "Don add #{START_TOKEN}/#{END_TOKEN} to sentences" do
        $options[:start_end_tokens] = false
    end

    opts.on '-v', '--verify [FILE]', 'Verify file integrity' do |file|
        $options[:verify] = file || true
    end

    opts.on '', '--dump-sequences', 'Dump data sequences',
            '(With --verify mode)' do
        $options[:dump_sentences] = true
    end

    opts.on( '-h', '--help', 'Display this screen' ) do
        puts opts
        exit
    end

end

optparse.banner << " WORDS_FILE[, WORDS_FILE]"
optparse.parse!

filenames = ARGV
if !$options[:verify]
    if filenames.length.zero?
        puts optparse
        exit 1
    end
end

if filenames.length > 0
    $sentences = []
    fidx = -1
    labels = $options[:labels] || []
    lbl_length = labels.length
    has_labels = lbl_length > 0

    last_label = nil
    file_count = 0
    all_files = filenames.map{|filename|
        filename = File.expand_path filename if filename[/^\~/]
        if filename['*']
            files = Dir.glob filename
        else
            if !File.exists? filename
                STDERR.puts "#{filename} not found!"
                exit 1
            end
            files = [filename]
        end
        fidx += 1
        if lbl_length > 0
            lbl_idx = fidx % lbl_length
            lbl = labels[lbl_idx]
            if (m = lbl.match(/^\[([\d\.,]+)\]$/))
                lbl = m[1]
                lbl << ',' if !lbl[',']
            end
            lbl = lbl.split(',').map{|l| l.to_f} if lbl[',']
            lbl = lbl.to_f if !lbl.is_a? Array
            if last_label.is_a?(Array)
                if !lbl.is_a?(Array)
                    STDERR.puts "ERROR: different label types"
                    exit 1
                elsif lbl.length != last_label.length
                    STDERR.puts "ERROR: different label lengths"
                    exit 1
                end
            elsif last_label.is_a? Array
                STDERR.puts "ERROR: different label types"
                exit 1
            end
            last_label = lbl
        end
        file_count += files.length
        {files: files, label: lbl}
    }.compact
    if file_count == 0
        STDERR.puts "ERROR: no files found!"
        exit 1
    end

    processed_file_count = 0
    max_sentences = $options[:max_sentences] || MAX_SENTENCES
    puts "Max Sentences: #{max_sentences}"
    max_sentences /= file_count
    start_end_tokens  = $options[:start_end_tokens] != false
    all_files.each_with_index{|f, idx|
        files = f[:files]
        label = f[:label]
        files.each_with_index{|filename, fidx|
            max_s = (max_sentences * (1 + processed_file_count))
            load_data_from filename, label: label,
                                     max_sentences: max_s,
                                     whole_file: $options[:whole_file],
                                     mode: $options[:mode],
                                     start_end_tokens: start_end_tokens
            processed_file_count += 1
            break if max_s > 0 && $sentences.length > max_s
        }
    }
    processed_perc = ((processed_file_count / file_count) * 100).round
    puts "File count: #{file_count}"
    puts "Processed Files: #{processed_file_count} (#{processed_perc}%)"
    puts "Sentences: #{$sentences.length}"

    #p $training_data[0]
    #p one_hot($training_data[0][0])
    $training_data ||= []
    tot_data_len = $training_data.length
    train_data_len = 0
    eval_data_len = 0
    test_data_len = 0
    train_data = []
    eval_data = []
    test_data = []
    if $training_label_offsets
        # equally distribute labels among different datasets
        slices = []
        $training_label_offsets.each{|lbl, offsets|
            offsets.each{|from, to|
                olen = (to + 1) - from
                eval_chunk_len = (olen * EVAL_DATALEN_PERC).to_i
                test_chunk_len = (olen * TEST_DATALEN_PERC).to_i
                chunk_len = eval_chunk_len + test_chunk_len
                if chunk_len > 0
                    cidx = (to + 1) - chunk_len
                    chunk = $training_data[cidx, chunk_len]
                    slices << [cidx, chunk_len]
                    eval_chunk = chunk.shift(eval_chunk_len)
                    test_chunk = chunk
                    eval_data += eval_chunk
                    test_data += test_chunk
                end
            }
        }
        slices.each{|slice|
            from, slen = slice
            $training_data.slice! from, slen
        }
        train_data = $training_data
        if $options[:shuffle]
            puts "Shuffling..."
            train_data.shuffle!
        end
        train_data_len = train_data.length
        eval_data_len = eval_data.length
        test_data_len = test_data.length
    else
        if $options[:shuffle]
            puts "Shuffling..."
            $training_data.shuffle!
        end
        eval_data_len = (tot_data_len * EVAL_DATALEN_PERC).to_i
        test_data_len = (tot_data_len * TEST_DATALEN_PERC).to_i
        if !$options[:same_dataset]
            train_data_len = tot_data_len - eval_data_len - test_data_len
            train_data = $training_data[0, train_data_len]
            eval_data_offset = train_data_len
        else
            train_data_len = tot_data_len
            train_data = $training_data
            eval_data_offset = tot_data_len - eval_data_len - test_data_len
        end
        eval_data = $training_data[eval_data_offset, eval_data_len]
        test_data = $training_data[eval_data_offset + eval_data_len,
                                   test_data_len]
    end
    if $labels_stats && $labels_stats.length > 0
        puts "Labels:"
        $labels_stats.each{|lbl, lblcount|
            perc = ((lblcount.to_f / tot_data_len) * 100).round
            puts " - #{lblcount} (#{perc}%) #{lbl.inspect}"
        }
    end

    ylen = nil
    train_c_data = train_data.map{|d|
        x, y = d
        len = x.length
        if has_labels
            if !ylen
                ylen = y.length
            elsif ylen != y.length
                STDERR.puts "ERROR: y length != from previous one: " +
                            "#{y.length} != #{ylen}"
                exit 1
            end
        end
        ([len.to_f] + x + y)
    }.flatten
    train_c_data_len = train_c_data.length + 1

    eval_c_data = eval_data.map{|d|
        x, y = d
        if ylen && y.length != ylen
            STDERR.puts "ERROR: eval y length != train y length: " +
                        "#{y.length} != #{ylen}"
            exit 1
        end
        len = x.length
        ([len.to_f] + x + y)
    }.flatten
    eval_c_data_len = eval_c_data.length + 1

    test_c_data = test_data.map{|d|
        x, y = d
        if ylen && y.length != ylen
            STDERR.puts "ERROR: test y length != train y length: " +
                        "#{y.length} != #{ylen}"
            exit 1
        end
        len = x.length
        ([len.to_f] + x + y)
    }.flatten
    test_c_data_len = test_c_data.length + 1

    train_c_data = "#{train_data_len.to_f},#{train_c_data.join(',')}"
    eval_c_data = "#{eval_data_len.to_f},#{eval_c_data.join(',')}"
    test_c_data = "#{test_data_len.to_f},#{test_c_data.join(',')}"
    words_c = $indexed_words.map{|w|
        w = w.encode(
            Encoding.find('ASCII'), invalid: :replace, undef: :replace,
            replace: '', universal_newline: true
        )
        w.inspect
    }
    output = $options[:output]
    output ||= '/tmp/w2v_training_data.h'
    h_name = "__#{File.basename(output).upcase.gsub(/[[:punct:]\s]+/, '_')}"
    c_code = <<-EOS
#ifndef #{h_name}
#define #{h_name}
#define VOCABULARY_SIZE     #{$vocabulary.length}
#define SENTENCE_COUNT      #{tot_data_len}
#define TRAIN_DATA_LEN      #{train_c_data_len}
#define EVAL_DATA_LEN       #{eval_c_data_len}
#define TEST_DATA_LEN       #{test_c_data_len}
#define TRAIN_SENTENCES     #{train_data_len}
#define EVAL_SENTENCES      #{eval_data_len}
#define TEST_SENTENCES      #{test_data_len}
#define Y_SEQUENCE_COUNT    #{ylen || 0}
char * #{$options[:words_var]}[] = {#{words_c.join(',')}};
PSFloat #{$options[:train_data_var]}[] = {#{train_c_data}};
PSFloat #{$options[:eval_data_var]}[] = {#{eval_c_data}};
PSFloat #{$options[:test_data_var]}[] = {#{test_c_data}};
#endif
    EOS
    File.open(output, 'w'){|f| f.write(c_code)}
    puts "\nWritten to #{output}"
    if $options[:verify]
        do_verify c_code, dump_sentences: $options[:dump_sentences]
    end
elsif $options[:verify]
    do_verify File.read($options[:verify]),
              dump_sentences: $options[:dump_sentences]
end
