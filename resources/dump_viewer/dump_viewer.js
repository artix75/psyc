const CELL_SIZE = 16;
const DEFAULT_PLAY_INTERVAL_SEC = 1000;
const MAX_SPEED = 15;
const MIN_SPEED = 0.25;

var CurrentViewer = null;
class DumpViewer {
    
    constructor (dump) {
        this.dump = dump;
        this.network = dump.network;
        if (!this.network) {
            alert("Missing 'network' if dump data");
            return;
        }
        this.$header = document.querySelector('header');
        this.$main = document.querySelector('main');
        let name = this.network.name || 'Unnamed Network';
        window.title = `Psyc Dump Viewer - ${name}`;
        document.querySelectorAll('.network-name').forEach(elem => {
            elem.innerText = this.network.name || 'Unnamed Network';
        });
        this.$networkInfo = document.querySelector('#network-info');
        this.$layerViews = document.querySelector('#layer-views');
        this.$weightViews = document.querySelector('#weight-views');
        this.$controls = document.querySelector('#controls');
        this.$forward = this.$controls.querySelector('.forward');
        this.$backward = this.$controls.querySelector('.backward');
        this.$play = this.$controls.querySelector('.play');
        this.$pause = this.$controls.querySelector('.pause');
        this.$stepForward = this.$controls.querySelector('.step-forward');
        this.$stepBackward = this.$controls.querySelector('.step-backward');
        this.$backward.classList.add('disabled');
        this.$stepBackward.classList.add('disabled');
        this.$stepInfo = this.$controls.querySelector('#step-info');
        this.$stepFunc = this.$stepInfo.querySelector('.step-func');
        this.$stepLayer = this.$stepInfo.querySelector('.step-layer');
        this.$stepFeat = this.$stepInfo.querySelector('.step-feat');
        this.$stepNeuron = this.$stepInfo.querySelector('.step-neuron');
        this.$speedControls = this.$controls.querySelector('#speed-controls');
        this.$increaseSpeed = this.$speedControls.querySelector('.increase');
        this.$decreaseSpeed = this.$speedControls.querySelector('.decrease');
        this.$currentSpeed =
            this.$speedControls.querySelector('.current-speed');
        this.renderNetworkInfo();
        this.renderLayerViews();
        this.currentStepIndex = -1;
        this.playing = false;
        this.playIntervalSec = DEFAULT_PLAY_INTERVAL_SEC;
        this.$forward.addEventListener('click', (event) => {
            this.$forward.classList.add('disabled');
            try {
                this.runNextStep();
            } catch (e) {
                console.error(e);
            }
            this.$forward.classList.remove('disabled');
        });
        this.$backward.addEventListener('click', (event) => {
            this.$backward.classList.add('disabled');
            try {
                this.runPreviousStep();
            } catch (e) {
                console.error(e);
            }
            this.$backward.classList.remove('disabled');
        });
        this.$stepForward.addEventListener('click', (event) => {
            this.$stepForward.classList.add('disabled');
            try {
                this.runNextStepMapPoint();
            } catch (e) {
                console.error(e);
            }
            this.$stepForward.classList.remove('disabled');
        });
        this.$stepBackward.addEventListener('click', (event) => {
            this.$stepBackward.classList.add('disabled');
            try {
                this.runPreviousStepMapPoint();
            } catch (e) {
                console.error(e);
            }
            this.$stepBackward.classList.remove('disabled');
        });
        this.$play.addEventListener('click', (event) => {
            this.$play.classList.add('disabled');
            try {
                this.play();
            } catch (e) {
                console.error(e);
            }
            this.$play.classList.remove('disabled');
        });
        this.$pause.addEventListener('click', (event) => {
            this.$pause.classList.add('disabled');
            try {
                this.pause();
            } catch (e) {
                console.error(e);
            }
            this.$pause.classList.remove('disabled');
        });
        this.$increaseSpeed.addEventListener('click', event => {
            this.increaseSpeed();
        });
        this.$decreaseSpeed.addEventListener('click', event => {
            this.decreaseSpeed();
        });
        this.runNextStep();
    }

    renderNetworkInfo() {
        Object.keys(this.network).forEach(key => {
            let val = this.network[key];
            if (typeof(val) === 'object') return;
            if (val instanceof Array) return;
            val = val.toString();
            let label = key.split(/_+/).map(w => {
                if (w.length > 0) {
                    let firstChar = w.charAt(0).toUpperCase();
                    if (w.length > 1) {
                        w = firstChar + w.substr(1).toLowerCase();
                    } else w = firstChar;
                }
                return w;
            }).join(' ');
            let $row = this.createInfoRow(label, val);
            this.$networkInfo.appendChild($row);
        }); 
        let header_h = this.$header.getBoundingClientRect().height;
        this.$main.style.marginTop = `${header_h}px`;
    }

    renderLayerViews() {
        let layers = this.dump.layer;
        if (!layers) {
            alert("Missing 'layers' if dump data");
            return;
        }
        let err = null;
        layers.forEach((layer, i) => {
            if (err) return;
            let index = layer.index;
            if (index === undefined) index = i;
            let type = layer.type;
            let $view = document.createElement('div');
            let cls = ['layer-view', type];
            $view.className = cls.join(' ');
            $view.setAttribute('data-index', index);
            $view.setAttribute('data-array-index', i);
            $view.setAttribute('data-type', type);
            let $title = document.createElement('h2');
            $title.className = 'layer-title';
            $title.innerText = `Layer ${index}: ${type}`,
            $view.appendChild($title);
            let features = layer.features || 1, fidx;
            if (features < 1) features = 1;
            let size = layer.size;
            if (!size) {
                err = `Missing size in layer ${i}`;
                return;
            }
            let feature_size = size / features;
            let nextLayer = layers[i + 1];
            let width = null;
            let padding = 0;
            let output_size = layer.output_size;
            if (nextLayer) {
                let input_size = nextLayer.input_size;
                if (input_size) {
                    input_size = input_size.split('x');
                    let w = parseInt(input_size[0]);
                    if (!isNaN(w) && w) width = w;
                }
                padding = nextLayer.padding || 0;
                if (padding < 0) padding = 0;
            }
            if (!width && output_size) {
                output_size = output_size.split('x');
                let w = parseInt(output_size[0]);
                if (!isNaN(w) && w) width = w;
            }
            /*if (!width) width = Math.sqrt(feature_size);*/
            if (!width) width = feature_size;
            if (width > 32) width = 1;
            let feature_rows = feature_size / width;
            let real_rows = feature_rows;
            let real_width = width;
            if (padding > 0) {
                real_rows += (padding * 2);
                real_width += (padding * 2);
            } 
            let region = layer.region;
            if (region) region = region.split('x').map(rs => parseInt(rs));
            let weights_size = null, feature_weights_size = null,
                prev_features = null;
            if (i > 0 && type !== 'Pooling') {
                let prevl = layers[i - 1];
                prev_features = (prevl.features || 1);
                if (prevl && region) {
                    let rw = region[0];
                    let rh = region[1] || rw;
                    weights_size = (rw * rh * prev_features);
                } else if (prevl && prevl.size) {
                    weights_size = prevl.size * prev_features;
                }
            }
            let $layout = document.createElement('table');
            $layout.className = 'layer-layout';
            $layout.style.width = `${real_width * CELL_SIZE}px`;
            $layout.style.height = `${real_rows * CELL_SIZE}px`;
            let neurod_idx = 0;
            for (fidx = 0; fidx < features; fidx++) {
                let $tbody = document.createElement('tbody');
                $tbody.setAttribute('data-feature-idx', fidx);
                let $thead = document.createElement('tr');
                $thead.className = 'feature-header';
                let $th = document.createElement('th');
                $th.setAttribute('colspan', real_width);
                $th.innerText = `Feature ${fidx}`;
                $thead.appendChild($th);
                $tbody.appendChild($thead);
                let r = 0;
                for (; r < real_rows; r++) {
                    let y = r - padding;
                    let $row = document.createElement('tr');
                    let cls = ['layer-row'];
                    if (r === 0 && fidx > 0) cls.push('first-feature-row');
                    let is_padding = (y < 0 || y >= feature_rows);
                    if (is_padding) cls.push('padding');
                    $row.className = cls.join(' ');
                    $row.setAttribute('data-feature', fidx);
                    $row.setAttribute('data-row', r);
                    $row.setAttribute('data-y', y);
                    let col = 0;
                    for (; col < real_width; col++) {
                        let x = col - padding;
                        let $neuron = document.createElement('td');
                        let is_padding = (
                            (y < 0 || y >= feature_rows) ||
                            (x < 0 || x >= width)
                        );
                        let cls = ['neuron'];
                        if (is_padding) cls.push('padding');
                        if (y === 0 && !is_padding) cls.push('first-row');
                        if (x === 0 && !is_padding) cls.push('first-col');
                        $neuron.className = cls.join(' ');
                        $neuron.innerHTML = '&nbsp;';
                        $neuron.style.width = ((1 / real_width) * 100) + '%';
                        $neuron.style.height = ((1 / real_width) * 100) + '%';
                        $neuron.setAttribute('data-x', x);
                        $neuron.setAttribute('data-y', y);
                        $neuron.setAttribute('data-coord', `${x},${y}`);
                        $row.appendChild($neuron);
                        if (!is_padding) {
                            let id = [index];
                            if (features > 1) id.push(fidx);
                            id.push(neurod_idx);
                            let realID = id.join('-');
                            id = 'neuron-' + id.join('-');
                            $neuron.setAttribute('id', id);
                            $neuron.setAttribute('data-index', neurod_idx);
                            $neuron.setAttribute('title', `Neuron ${realID}`);
                            neurod_idx++;
                            $neuron.classList.add('real-neuron');
                            $neuron.addEventListener('click', event => {
                                let step_idx =
                                    this.findNearestStepIndexForNeuron(realID);
                                if (!step_idx) return;
                                if (step_idx === this.currentStepIndex)
                                    return;
                                let step = this.dump.step[step_idx];
                                if (!step) return;
                                this.currentStepIndex = step_idx;
                                this.runStep(step, step_idx);
                            });
                        } else {
                            if (x === -1 && y >= 0 && y < feature_rows)
                                $neuron.classList.add('before-area');
                            if (y === -1 && x >= 0 && x < width)
                                $neuron.classList.add('above-area');
                        }
                    }
                    $tbody.appendChild($row);
                }
                $layout.appendChild($tbody);
            }
            let $container = document.createElement('div');
            $container.className = 'layer-container';
            let h = parseInt($layout.style.height);
            this.maxLayerHeight = this.maxLayerHeight || 0;
            //h /= features;
            if (h > this.maxLayerHeight) this.maxLayerHeight = h;
            else if (this.maxLayerHeight > 0) h = this.maxLayerHeight;
            $container.style.height = `${h}px`;
            $container.style.maxHeight = `${h}px`;
            $container.appendChild($layout);
            $view.appendChild($container);
            this.$layerViews.appendChild($view);
            /* Weights */
            if (weights_size && prev_features) {
                let feature_weight_size = weights_size / prev_features;
                let width = null, rows = null;
                if (region) {
                    width = region[0];
                    rows = region[1] || width;
                    if ((width * rows) !== feature_weight_size) {
                        alert(`Layer ${i} region size ${layer.region} = ` +
                              `${width * rows} doesn't match with ` +
                              `feature weight size ${feature_weight_size}`);
                        return;
                    }
                }
                if (!width) width = weights_size;
                if (width > 32) width = 1;
                rows = weights_size / width;
                let $weight_view = document.createElement('div');
                $weight_view.className = 'weight-view';
                let $weight_container = document.createElement('div');
                $weight_container.className = 'weight-container';
                let $weight_layout = document.createElement('table');
                $weight_container.appendChild($weight_layout);
                $weight_container.style.height = `${h}px`;
                $weight_container.style.maxHeight = `${h}px`;
                let $h2 = document.createElement('h2');
                $h2.innerText = `Layer ${i} Weights`;
                $weight_view.appendChild($h2);
                $weight_view.appendChild($weight_container);
                $weight_view.setAttribute('id', `layer-${i}-weights`);
                $weight_view.setAttribute('data-layer', i);
                $weight_layout.className = 'weights';
                $weight_layout.setAttribute('data-layer', i);
                $weight_layout.setAttribute('data-previous-layer', i - 1);
                $weight_layout.style.width = `${width * CELL_SIZE}px`;
                $weight_layout.style.height = `${rows * CELL_SIZE}px`;
                let $tbody = document.createElement('tbody');
                let widx = 0, r, x;
                for (r = 0; r < rows; r++) {
                    let $tr = document.createElement('tr');
                    for (x = 0; x < width; x++) {
                        let $w = document.createElement('td');
                        $w.className = 'weight';
                        let windex = widx++;
                        if ((windex % feature_weight_size) === 0)
                            $tr.classList.add('first-feature-row');
                        $w.setAttribute('id', `weight-${windex}`);
                        $w.setAttribute('title', `Weight ${windex}`);
                        $w.innerHTML = '&nbsp;';
                        $w.style.width = ((1 / width) * 100) + '%';
                        $w.style.height = ((1 / rows) * 100) + '%';
                        $tr.appendChild($w);
                    }
                    $tbody.appendChild($tr);
                }
                $weight_layout.appendChild($tbody);
                this.$weightViews.appendChild($weight_view);
            }
        });
        if (err) {
            alert(err);
            throw new Error(err);
            return;
        }
    }

    runStep(step, step_idx) {
        if (step_idx === undefined) step_idx = this.dump.step.indexOf(step);
        let func = step.func;
        let layer = step.layer;
        this.$stepFunc.innerText = func || '---';
        if (isNaN(layer)) this.$stepLayer.innerText = '---';
        else this.$stepLayer.innerText = `Layer: ${layer}`;
        let neuron_id = step.neuron;
        let ninfo = {};
        if (neuron_id) ninfo = this.parseNeuronID(neuron_id) || {};
        this.$stepFeat.innerText = `Feature: ${ninfo.feature || 0}`;    
        this.$stepNeuron.innerText = `Neuron: ${ninfo.index || 0}`;
        if (step_idx === 0) {
            this.$backward.classList.add('disabled');
            this.$stepBackward.classList.add('disabled');
            this.$forward.classList.remove('disabled');
            this.$stepForward.classList.remove('disabled');
        } else if (step_idx === this.dump.step.length - 1) {
            this.$backward.classList.remove('disabled');
            this.$stepBackward.classList.remove('disabled');
            this.$forward.classList.add('disabled');
            this.$stepForward.classList.add('disabled');
        } else {
            this.$backward.classList.remove('disabled');
            this.$forward.classList.remove('disabled');
            let map_idx = this.getCurrentStepMapIndex();
            if (map_idx === 0) {
                this.$stepBackward.classList.add('disabled');
                this.$stepForward.classList.remove('disabled');
            } else if (map_idx === this.dump.step_map.length - 1) {
                this.$stepBackward.classList.remove('disabled');
                this.$stepForward.classList.add('disabled');
            } else {
                this.$stepBackward.classList.remove('disabled');
                this.$stepForward.classList.remove('disabled');
            }
        }
        if (step.phase === 'feedforward')
            this.runFeedforwardStep(step, step_idx);
        else if (step.phase === 'backprop')
            this.runBackpropStep(step, step_idx);
    }

    runNextStep() {
        let step_idx = ++this.currentStepIndex;
        let step = this.dump.step[step_idx];
        if (!step) {
            if (this.currentStepIndex >= this.dump.step.length)
                this.currentStepIndex = this.dump.step.length - 1;
            return;
        }
        this.runStep(step, step_idx);
    }

    runPreviousStep() {
        let step_idx = --this.currentStepIndex;
        if (step_idx < 0) {
            this.currentStepIndex = 0;
            return;
        }
        let step = this.dump.step[step_idx];
        if (!step) return;
        this.runStep(step, step_idx);
    }

    runNextStepMapPoint() {
        let cur_point_idx = this.getCurrentStepMapIndex();
        if (cur_point_idx < 0) return;
        let point = this.dump.step_map[++cur_point_idx];
        if (point) {
            let step_idx = point.step_index;
            let step = this.dump.step[step_idx];
            if (step) {
                this.currentStepIndex = step_idx;
                this.runStep(step, step_idx);
            }
        }
    }

    runPreviousStepMapPoint() {
        let cur_point_idx = this.getCurrentStepMapIndex();
        if (--cur_point_idx < 0) return;
        let point = this.dump.step_map[cur_point_idx];
        if (point) {
            let step_idx = point.step_index;
            let step = this.dump.step[step_idx];
            if (step) {
                this.currentStepIndex = step_idx;
                this.runStep(step, step_idx);
            }
        }
    }

    playStep() {
        if (!this.playing) {
            this.$controls.classList.remove('playing');
            this.playTimeout = null;
            return false;
        }
        this.playing = true;
        this.$controls.classList.add('playing');
        this.runNextStep();
        if (this.currentStepIndex >= this.dump.step.length) {
            this.$controls.classList.remove('playing');
            this.playTimeout = null;
            this.playing = false;
            return false;
        }
        return setTimeout(() => {
            this.playStep();
        }, this.playIntervalSec);
    }

    play() {
        if (this.playing) return;
        this.playing = true;
        this.playTimeout = this.playStep();
    }

    pause() {
        if (!this.playing) return;
        this.playing = false;
        if (this.playTimeout) clearTimeout(this.playTimeout);
        this.playTimeout = null;
        this.$controls.classList.remove('playing');
    }

    increaseSpeed() {
        if (this.currentSpeed === undefined) this.currentSpeed = 1;
        if (this.currentSpeed >= MAX_SPEED) return;
        let incr_by = 1;
        if (this.currentSpeed < 1) incr_by = 0.25;
        this.currentSpeed += incr_by;
        this.playIntervalSec = 1000 / this.currentSpeed;
        this.$currentSpeed.innerText = `${this.currentSpeed}x`;
        if (this.currentSpeed === MAX_SPEED)
            this.$increaseSpeed.classList.add('disabled');
        else this.$increaseSpeed.classList.remove('disabled');
    }

    decreaseSpeed() {
        if (this.currentSpeed === undefined) this.currentSpeed = 1;
        if (this.currentSpeed <= MIN_SPEED) return;
        let decr_by = 1;
        if (this.currentSpeed <= 1) decr_by = 0.25;
        this.currentSpeed -= decr_by;
        this.playIntervalSec = 1000 / this.currentSpeed;
        this.$currentSpeed.innerText = `${this.currentSpeed}x`;
        if (this.currentSpeed === MIN_SPEED)
            this.$decreaseSpeed.classList.add('disabled');
        else this.$decreaseSpeed.classList.remove('disabled');
    }

    runFeedforwardStep(step, step_idx) {
        let layer_index = step.layer;
        if (layer_index === undefined) {
            alert(`Missing layer in step ${step_idx}`);
            return;
        }
        let layer = this.dump.layer[layer_index];
        if (!layer) {
            alert(`Invalid layer ${layer_index} in step ${step_idx}`);
            return;
        }
        let neuron_id = step.neuron;
        if (!neuron_id) {
            alert(`Missing neuron in step ${step_idx}`);
            return;
        }
        let prev_neuron_id = step.previous_neuron;
        if (!prev_neuron_id) {
            alert(`Missing previous_neuron in step ${step_idx}`);
            return;
        }
        let $n = this.getNeuronCellByID(neuron_id);
        if (!$n) {
            alert(
                `Neuron cell not found for ID ${neuron_id} in step ${step_idx}`
            );
            return;
        }
        let $prev_n = this.getNeuronCellByID(prev_neuron_id);
        if (!$prev_n) {
            alert(
                `Neuron cell not found for ID ${prev_neuron_id} in ` +
                `step ${step_idx}`
            );
            return;
        }
        this.displayLayerWeights(layer_index);
        this.resetNeuronsState();
        $n.classList.add('current');
        $prev_n.classList.add('target');
        $n.scrollIntoView({block: 'center', inline: 'nearest'});
        $prev_n.scrollIntoView({block: 'center', inline: 'nearest'});
        let region = step.region;
        if (region) {
            if (region.length < 4) {
                alert(
                    `Invalid region ${region.join(',')} in ` +
                    `step ${step_idx}`
                );
                return;
            }
            let x1 = region[0], y1 = region[1],
                x2 = region[2], y2 = region[3];
            let y, x;
            let feature = 0;
            let ninfo = this.parseNeuronID(prev_neuron_id);
            if (ninfo.feature !== undefined) feature = ninfo.feature;
            for (y = y1; y < y2; y++) {
                for (x = x1; x < x2; x++) {
                    let $cell = this.getRegionCell(
                        layer_index - 1, feature, x, y
                    );
                    if (!$cell) break;
                    $cell.classList.add('in-region');
                }
            }
        }
        let widx = step.weight_idx;
        if (widx === undefined) widx = step.weight_index;
        if (widx !== undefined)
            this.setWeightState(layer_index, widx, 'target');
    }

    runBackpropStep(step, step_idx) {
        let layer_index = step.layer;
        if (layer_index === undefined) {
            alert(`Missing layer in step ${step_idx}`);
            return;
        }
        let layer = this.dump.layer[layer_index];
        if (!layer) {
            alert(`Invalid layer ${layer_index} in step ${step_idx}`);
            return;
        }
        let neuron_id = step.neuron;
        if (!neuron_id) {
            alert(`Missing neuron in step ${step_idx}`);
            return;
        }
        let $n = this.getNeuronCellByID(neuron_id);
        if (!$n) {
            alert(
                `Neuron cell not found for ID ${neuron_id} in step ${step_idx}`
            );
            return;
        }
        let next_neuron_id = step.next_neuron;
        let prev_neuron_id = step.previous_neuron;
        let $next_n = null, $prev_n = null;
        if (!next_neuron_id && !prev_neuron_id) {
            alert(`Missing next_neuron and prev_neuron in step ${step_idx}`);
            return;
        }
        let weight_layer_idx = layer_index, weight_state = 'target';
        if (next_neuron_id) {
            $next_n = this.getNeuronCellByID(next_neuron_id);
            if (!$next_n) {
                alert(
                    `Neuron cell not found for ID ${next_neuron_id} in ` +
                    `step ${step_idx}`
                );
                return;
            }
            let next_neuron_info = this.parseNeuronID(next_neuron_id);
            if (next_neuron_info) weight_layer_idx = next_neuron_info.layer;
            weight_state = 'source';
        }
        if (prev_neuron_id) {
            $prev_n = this.getNeuronCellByID(prev_neuron_id);
            if (!$prev_n) {
                alert(
                    `Neuron cell not found for ID ${prev_neuron_id} in ` +
                    `step ${step_idx}`
                );
                return;
            }
        }
        this.displayLayerWeights(weight_layer_idx);
        this.resetNeuronsState();
        $n.classList.add('current');
        if ($next_n) $next_n.classList.add('source');
        if ($prev_n) $prev_n.classList.add('target');
        $n.scrollIntoView({block: 'center', inline: 'nearest'});
        if ($next_n)
            $next_n.scrollIntoView({block: 'center', inline: 'nearest'});
        if ($prev_n)
            $prev_n.scrollIntoView({block: 'center', inline: 'nearest'});
        let region = step.region;
        if (region) {
            if (region.length < 4) {
                alert(
                    `Invalid region ${region.join(',')} in ` +
                    `step ${step_idx}`
                );
                return;
            }
            let x1 = region[0], y1 = region[1],
                x2 = region[2], y2 = region[3];
            let y, x;
            let feature = 0;
            let ref_neuron_id = (prev_neuron_id || neuron_id);
            /*let lidx = (ref_neuron_id === next_neuron_id ? layer_index + 1 :
                                                           layer_index - 1);*/
            let ninfo = this.parseNeuronID(ref_neuron_id);
            if (ninfo.feature !== undefined) feature = ninfo.feature;
            for (y = y1; y < y2; y++) {
                for (x = x1; x < x2; x++) {
                    let $cell = this.getRegionCell(
                        ninfo.layer, feature, x, y
                    );
                    if (!$cell) break;
                    $cell.classList.add('in-region');
                }
            }
        }
        let widx = step.weight_idx;
        if (widx === undefined) widx = step.weight_index;
        if (widx !== undefined)
            this.setWeightState(weight_layer_idx, widx, weight_state);
    }

    displayLayerWeights(layer) {
        let layer_idx = null;
        if (typeof(layer) === 'number') layer_idx = layer;
        else layer_idx = layer.index;
        if (layer_idx === null || layer_idx === undefined) {
            alert(`displayLayerWeights: Invalid layer ${layer}`);
            return;
        }
        layer_idx = parseInt(layer_idx);
        if (!this.$weightViewElements) {
            this.$weightViewElements =
                document.querySelectorAll('.weight-view');
        }
        this.$weightViewElements.forEach($view => {
            let lidx = $view.getAttribute('data-layer');
            if (lidx === null || lidx === undefined) return;
            lidx = parseInt(lidx);
            if (isNaN(lidx)) return;
            if (lidx === layer_idx) $view.classList.add('current');
            else $view.classList.remove('current');
        });
    }

    parseNeuronID(id) {
        if (!this.neuronIDCache) this.neuronIDCache = {};
        let info = this.neuronIDCache[id];
        if (info) return info;
        let components = id.split('-');
        info = {};
        if (components.length === 0) return info;
        info.layer = parseInt(components[0]);
        if (components.length >= 3) {
           info.feature = parseInt(components[1]) 
           info.index = parseInt(components[2]) 
        } else if (components.length === 2) {
           info.index = parseInt(components[1]) 
        }
        this.neuronIDCache[id] = info;
        return info;
    }

    getRegionCell(layer, feature, x, y) {
        let index = (typeof(layer) === 'number' ? layer : layer.index);
        if (isNaN(index)) {
            alert(`Invalid layer ${layer}`);
            return null;
        }
        let coord = [x, y].join(',');
        let sel =
            `.layer-view[data-index="${index}"] ` +
            `tr[data-feature="${feature}"] .neuron[data-coord="${coord}"]`;
        return document.querySelector(sel);
    }

    getNeuronCellByID(id) {
        return document.querySelector(`#neuron-${id}`);
    }

    resetNeuronsState() {
        document.querySelectorAll('.neuron.current').forEach($n => {
            $n.classList.remove('current');
        });
        document.querySelectorAll('.neuron.target').forEach($n => {
            $n.classList.remove('target');
        });
        document.querySelectorAll('.neuron.source').forEach($n => {
            $n.classList.remove('source');
        });
        document.querySelectorAll('.neuron.in-region').forEach($n => {
            $n.classList.remove('in-region');
        });
        document.querySelectorAll('.weight.source').forEach($n => {
            $n.classList.remove('source');
        });
        document.querySelectorAll('.weight.target').forEach($n => {
            $n.classList.remove('target');
        });
    }

    createInfoRow(label, value) {
        let $row = document.createElement('li');
        let $label = document.createElement('label');
        let $value = document.createElement('span');
        $value.className = 'value';
        $label.innerText = label;
        $value.innerText = value || '';
        $row.appendChild($label);
        $row.appendChild($value);
        return $row;
    }

    getCurrentStepMapIndex() {
        if (!this.dump.step_map) return -1;
        let step_idx = this.currentStepIndex;
        if (step_idx < 0) step_idx = 0;
        let idx = -1, len = this.dump.step_map.length, i;
        for (i = 0; i < len; i++) {
            let point = this.dump.step_map[i];
            let next_i = i + 1;
            let nextPoint = (next_i < len ? this.dump.step_map[next_i] : null);
            let in_range = (
                step_idx >= point.step_index && !nextPoint ||
                nextPoint.step_index > step_idx
            );
            if (in_range) {
                idx = i;
                break;
            }
        }
        return idx;
    }

    getCurrentStep() {
        let step_idx = this.currentStepIndex;
        if (step_idx < 0) return null;
        return this.dump.step[step_idx] || null;
    }

    findNextStepIndexForNeuron(neuronID) {
        if (!neuronID) return null;
        let step_idx = this.currentStepIndex,
            step_len = this.dump.step.length, i;
        if (step_idx < 0) step_idx = 0;

        for (i = step_idx; i < step_len; i++) {
            let step = this.dump.step[i];
            let step_neuron_id = step.neuron;
            if (!step_neuron_id) continue;
            if (step_neuron_id === neuronID) return i;
        }
        return null;
    }

    findPreviousStepIndexForNeuron(neuronID) {
        if (!neuronID) return null;
        let step_idx = this.currentStepIndex, i;
        if (step_idx < 0) return null;
        for (i = step_idx; i >= 0; i--) {
            let step = this.dump.step[i];
            let step_neuron_id = step.neuron;
            if (!step_neuron_id) continue;
            if (step_neuron_id === neuronID) return i;
        }
        return null;
    }

    findNearestStepIndexForNeuron(neuronID) {
        if (!neuronID) return null;
        let neuron_info = this.parseNeuronID(neuronID);
        if (!neuron_info) return null;
        let step_idx = this.currentStepIndex, step_len = this.dump.step.length;
        if (step_idx <= 0) return this.findNextStepIndexForNeuron(neuronID);
        else if (step_idx >= step_len - 1) {
            return this.findPreviousStepIndexForNeuron(neuronID);
        }
        let current_step = this.dump.step[step_idx];
        if (!current_step) return null;
        let current_neuron_id = current_step.neuron;
        if (!current_neuron_id) return null;
        if (current_neuron_id === neuronID) return step_idx;
        let current_info = this.parseNeuronID(current_neuron_id);
        if (!current_info) return null;
        let direction = 0;
        if (neuron_info.layer < current_info.layer) direction = -1;
        else if (neuron_info.layer > current_info.layer) direction = 1;
        else if(neuron_info.feature !== undefined &&
                neuron_info.feature < current_info.feature) direction = -1;
        else if(neuron_info.feature !== undefined &&
                neuron_info.feature > current_info.feature) direction = 1;
        else if (neuron_info.index < current_info.index) direction = -1;
        else if (neuron_info.index > current_info.index) direction = 1;
        if (direction === 0) return this.findNextStepIndexForNeuron(neuronID);
        if (current_info.phase === 'backprop') direction *= -1;
        if (direction === 1) return this.findNextStepIndexForNeuron(neuronID);
        else return this.findPreviousStepIndexForNeuron(neuronID);
    }

    getWeightsTable(layer) {
        let layer_idx = null;
        if (typeof(layer) === 'number') layer_idx = layer;
        else layer_idx = layer.index;
        if (isNaN(layer_idx)) return null;
        return document.querySelector(`#layer-${layer_idx}-weights`);
    }

    setWeightState(layer, widx, state) {
        state = state || 'target';
        let $weight = null;
        let $weights = this.getWeightsTable(layer);
        if ($weights) $weight = $weights.querySelector(`#weight-${widx}`);
        if ($weight) {
            $weight.classList.add(state);
            $weight.scrollIntoView({block: 'center', inline: 'nearest'});
        }
    }

};

document.addEventListener("DOMContentLoaded", () => {
    if (typeof(DumpData) === 'undefined') {
        alert("No DumpData!");
        return;
    }
    console.log('Loaded!');
    let viewer = new DumpViewer(DumpData);
    CurrentViewer = viewer;
});
