# Oemer (End-to-end OMR)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BreezeWhite/oemer/blob/main/colab.ipynb)
[![PyPI version](https://badge.fury.io/py/oemer.svg)](https://badge.fury.io/py/oemer)
![PyPI - License](https://img.shields.io/github/license/BreezeWhite/oemer)
[![Downloads](https://img.shields.io/pypi/dm/oemer?color=orange)](https://pypistats.org/packages/oemer)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8429346.svg)](https://doi.org/10.5281/zenodo.8429346)



End-to-end Optical Music Recognition system build on top of deep learning models and machine learning techniques.
Able to transcribe on skewed and phone taken photos. The models were trained to identify *Western Music Notation*, which could mean the system will probably not work on transcribing hand-written scores or other notation types.


![](figures/tabi_mix.jpg)

https://user-images.githubusercontent.com/24308057/136168551-2e705c2d-8cf5-4063-826f-0e179f54c772.mp4

## Quick Start
``` bash
# Install from PyPi
pip install oemer

# (optional) Install the Tensorflow version.
pip install oemer[tf]

# (optional) Or install the newest updates directly from Github.
pip install git+https://github.com/BreezeWhite/oemer

# Run
oemer <path_to_image>
```

The `oemer` command will output the transcribed MusicXML file and an image of analyzed elements to current directory.

With GPU, this usually takes around 3~5 minutes to finish. For the first time running, the checkpoints will be downloaded automatically and may take up to 10 minutes to download, depending on your connection speed. Checkpoints can also be manually downloaded from [here](https://github.com/BreezeWhite/oemer/releases/tag/checkpoints). Put checkpoint files start with `1st_*` to `oemer/checkpoints/unet_big`, `2nd_*` to `oemer/checkpoints/seg_net`, and rename the files by removing the prefix `1st_`, `2nd_`.

Default to use **Onnxruntime** for inference. If you want to use **Tensorflow** for running the inference,
add `--use-tf` to the command and make sure there is TF installed.


## Issue

### IMPORTANT!!
***Please follow the issue template, fill in all required information. Otherwise, the issue will be closed directly without further processing!***

If you encounter errors, try adding `--without-deskew` first (see [issue #9](https://github.com/BreezeWhite/oemer/issues/9)). If the problem still exists, file an issue and make sure following the template format.

### Available options
```
usage: oemer [-h] [-o OUTPUT_PATH] [--use-tf] [--save-cache] [-d] img_path

End-to-end OMR command line tool. Receives an image as input, and outputs
MusicXML file.

positional arguments:
  img_path              Path to the image.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Path to output the result file. (default: ./)
  --use-tf              Use Tensorflow for model inference. Default is to use
                        Onnxruntime. (default: False)
  --save-cache          Save the model predictions and the next time won't
                        need to predict again. (default: False)
  -d, --without-deskew  Disable the deskewing step if you are sure the image
                        has no skew. (default: False)
```

## Citation

```
@software{yoyo_2023_8429346,
  author       = {Yoyo and
                  Christian Liebhardt and
                  Sayooj Samuel},
  title        = {BreezeWhite/oemer: v0.1.7},
  month        = oct,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.1.7},
  doi          = {10.5281/zenodo.8429346},
  url          = {https://doi.org/10.5281/zenodo.8429346}
}
```

## Technical Details

This section describes the detail techniques for solving the OMR problem. The overall flow can also be found in [oemer/ete.py](https://github.com/meteo-team/oemer/blob/main/oemer/ete.py), which is also the entrypoint for `oemer` command.

Notice that all descriptions below are simplfied compared to the actual implementations. Only core concepts are covered.

### Model Training

There are two UNet models being used: one serves to separate stafflines and all other symbols, and the other for separating more detailed symbol types (see [Model Prediction](#model-prediction) below). The training script is under `oemer/train.py`.

The two models use different datasets for training: [CvcMuscima-Distortions](http://pages.cvc.uab.es/cvcmuscima/index_database.html) for training the first model, and [DeepScores-extended](https://zenodo.org/record/4012193) for the second model. Both trainings leverage multiple types of image augmentation techniques to enhance the robustness (see [here](https://github.com/BreezeWhite/oemer/blob/main/oemer/train.py#L50-L108)).

To identify invidual symbol types on the predictions, SVM models are used. The data used to train SVM models are extracted from [DeepScores-extended](https://zenodo.org/record/4012193). There are three different SVM models that are used to classify symbols. More details can be found in [oemer/classifier.py](https://github.com/BreezeWhite/oemer/blob/main/oemer/classifier.py).

### Model Prediction
Oemer first predicts different informations with two image semantic segmentation models: one for
predicting stafflines and all other symbols; and the second model for more detailed symbol informations,
including noteheads, clefs, stems, rests, sharp, flat, natural.


<p align='center'>
    <img width="70%" src="figures/tabi_model1.jpg">
    <p align='center'>Model one for predicting stafflines (red) and all other symbols (blue).</p>
</p>
<p align='center'>
    <img width="70%" src="figures/tabi_model2.jpg">
    <p align='center'>Model two for predicting noteheads (green), clefs/sharp/flat/natural (pink), and stems/rests (blue).</p>
</p>

### Dewarping

Before proceed to recognizing symbols, it is necessary to deskew the photo first, since
later processes assume stafflines are all horizontally aligned, and also the position 
of noteheads, rests and all other things are all depending on this assumption.

For the dewarping process, it can be summarized imto six steps as shown in the below figure.

<p align='center'>
    <img width="100%" src="figures/dewarp_steps.png">
    <p align='center'>Steps to dewarp the curved image.</p>
</p>


The dewarping map will be applied to all the predicted informations produced by the two NN models.

### Staffline Extraction

After dewarping, stafflines are being parsed first. This step plays the most important role
during the whole process, as this is the foundation to all later steps. Ths most important information is 
`unit_size`, which is the interval between stafflines. It's obvious that all the size-related and
distance-related information in a music score, all relate to the interval size of stafflines.

Stafflines are processed part-by-part horizontally, as shown below:

<p align='center'>
    <img width="50%" src="figures/staffs.jpg">
</p>

For each part, the algorithm finds the lines by accumulating positive pixels by rows.
After summarizing the amounts for each row, we get the following statistics:

<p align='center'>
    <img width="50%" src="figures/staffline_peaks.png">
</p>

The algorithm then picks all the peaks and applies additional rules to filter out false positive peaks.
The final picked true positive peaks (stafflines) are marked with red dots.

Another important information is **tracks** and **groups**. For a conventional piano score, there are
two tracks, for left and right hand, respectively. The two tracks futher forms a group. For this information,
the algorithm uses the symbol predictions and parse the barline information to infer possible
grouping of tracks.

After extraction, the informations are stored into list of `Staff` instances. An example 
`Staff` instance representation is as following:

``` bash
# Example instance of oemer.staffline_extraction.Staff
Staff(
    Lines: 5  # Contains 5 stafflines.
    Center: 1835.3095048449181  # Y-center of this block of staff.
    Upper bound: 1806  # Upper bound of this block of staff (originated from left-top corner).
    Lower bound: 1865  # Lower bound of this block of staff (originated from left-top corner).
    Unit size: 14.282656749749265  # Average interval of stafflines.
    Track: 1  # For two-handed piano score, there are two tracks.
    Group: 3  # For two-handed piano score, two tracks are grouped into one.
    Is interpolation: False  # Is this block of staff information interpolated.
    Slope: -0.0005315575840202954  # Estimated slope
)
```

### Notehead Extraction

The next step is to extract noteheads, which is the second important information to be parsed.

Steps to extract noteheads are breifly illustrated in the following figure:

<p align='center'>
    <img width="100%" src="figures/notehead.png">
</p>


One of the output channel of the second model predicts the noteheads map, as can be seen in the
top-middle image. The algorithm then pre-processes it with morphing to refine the information.
Worth noticing here is that the model was trained to predict 'hollow' notes as solid noteheads,
which thus the empty noteheads won't be eliminated by the morphing.

Next, the algorithm detects the bounding box of each noteheads. Since the noteheads could
overlap with each other, the initial detection could contain more than one notehead. 
To deal with such situation, the algorithm integrates the information `unit_size` to approximate
how many noteheads are actually there, in both horizontal and vertical directions. The result
is shown in the bottom-left figure.

As we force the model to predict both half and whole notes to be solid noteheads, we need to
setup rules to decide whether they are actually half or whole notes. This could be done by
simply compare the region coverage rate between the prediction and the original image.
The result is shown in the bottom-middle figure.

Finally, the last thing to be parsed is the position of noteheads on stafflines. Index 0
originates from the bottom line space (D4 for treble clef, and F3 for bass clef), higher pitch
having larger index number. There could also be negative numbers. In this step, noteheads are also assigned with
track and group number, indicating which stave they belong to. The bottom-right figure shows
the result.


``` bash
# Example instance of oemer.notehead_extraction.NoteHead
Notehead 12 (  # The number refers to note ID
    Points: 123  # Number of pixels included in this notehead.
    Bounding box: [649 402 669 419] # xyxy
    Stem up: None  # Direction of the stem, will be infered in later steps.
    Track: 1  # For a two-hand piano score, this represents the left hand track.
    Group: 0  # The staring group of the score.
    Pitch: None  # Actual pitch in MIDI number, will be infered in later steps.
    Dot: False  # Whether the note contains a dot.
    Label: NoteType.HALF_OR_WHOLE  # Initial guess of the rhythm type.
    Staff line pos: 4  # Position on stafflines. Counting from D4 for treble clef.
    Is valid: True  # Flag for marking if the note prediction is valid.
    Note group ID: None  # Note group ID this note belongs to. Will be infered in later steps.
    Sharp/Flat/Natural: None  # Accidental type of this note. Will be infered in later steps.
)

```

### Note Group Extraction

This step groups individual noteheads into chords that should be played at the same time.

A quick snippet of the final result is shown below:

<p align='center'>
    <img width="80%" src="figures/note_group.png">
</p>

The first step is to group the noteheads according mainly to their distance vertically, and then
the overlapping and a small-allowed distance horizontally.

After the initial grouping, the next is to parse the stem direction and further use this
information to refine the grouping results. Since there could be noteheads that are vertically
very close, but have different directions of stems. This indicates that there are two
different melody lines happening at the same time. This is specifically being considered
in `oemer` and taken care of over all the system.

``` bash
# Example instance of oemer.note_group_extraction.NoteGroup
Note Group No. 0 / Group: 0 / Track: 0 :(
    Note count: 1
    Stem up: True
    Has stem: True
)
```

### Symbol Extraction

After noteheads being extracted, there remains other important musical annotations need
to be parsed, such as keys, accidentals, clefs, and rests.
As mentioned before, the second model predicts different pairs of symbols in the same channel
for the ease of training. Additional separation of the information is thus required.

#### Clefs & SFN
For the clefs/sfn (short for sharp, flat, natural) pair, the initial intention for grouping
them together, is that it's easier to distinguish the difference through their size and
the region coverage rate (tp_pixels / bounding_box_size). This is exactly what the
algorithm being implemented to recognize them. After the clef/sfn classification,
Further recognition leverages SVM models to classify them into the correct symbol
types (e.g. gclef, sharp, flat).

<p align='center'>
    <img width="80%" src="figures/clefs_sfns.png">
</p>

``` bash
# Example instance of oemer.symbol_extraction.Clef
Clef: F_CLEF / Track: 1 / Group: 1

# Example instance of oemer.symbol_extraction.Sfn
SFN: NATURAL / Note ID: 186 / Is key: False / Track: 0 / Group: 0
```

#### Barlines

Extracts barlines using both models' output. The algorithm first uses the second model's prediction,
the channel contains rests and 'stems' (which should be 'straight lines' actually). Since the
previous step while extracting note groups has already used the 'stem' information, so the rest
part of unused 'stems' should be barlines. However, due to some bugs of the training dataset,
the model always predicts barlines, that should be longer than stems, into the same length of
stems. It is thus the algorithm needs the first model's output to extract the 'actual' barlines
with real lengths. By overlapping the two different information, the algorithm can easily filter out
most of non-barline objects in the prediction map. Further extraction applies additional rules to
estimate barlines. The result can be seen as follow:

<p align='center'>
    <img width="80%" src="figures/barlines.png">
</p>

And the representation of a barline instance:
``` bash
# Example instance of oemer.symbol_extraction.Barline
Barline / Group: 3
```

There is no track information of barline since one barline is supposed to 
occupy multiple tracks.

#### Rests

Having used all the 'stems' information in the output channel during the last few
steps, the rest symbols should be 'rests'. List of rules are also applied to
filter the symbols. The recognition of the rest types are done by using trained SVM model.
As a result, above process outputs the following result:

<p align='center'>
    <img width="80%" src="figures/rests.png">
</p>

Representation of the rest instance:
``` bash
# Example instance of oemer.symbol_extraction.Rest
Rest: EIGHTH / Has dot: None / Track: 1 / Group: 1
```


### Rhythm Extraction

This is probably the most time consuming part except for the model inference.
There are two things that effect the rhythm: dot and beams/flags. The later two (beams, flags)
are considered the same thing in the extraction. In this step, model one's prediction
is used, including both channels (stafflines, symbols). This process updates attributes
in-place.

The algorithm first parse the information of dot for each note. The symbols map is first
subtracted by other prediction maps (e.g. stems, noteheads, clefs, etc.), and then use
the remaining part for scanning the dots. Since the region of a dot is small, the algorithm
morphs the map first. After amplifying the dot information, the algorithm scans a small region
nearby every detected noteheads, calculate the ratio of positive samples to the region, and
determine whether there is a dot by a given certain threshold.

<p align='center'>
    <img width="80%" src="figures/dots.png">
</p>

Here comes the most difficult and critical part amongst all steps, since rhythm hugely
influence the listening experience.
Few steps are included to extract beams/flags:
- Initial parsing
- Check overlapping with noteheads and stems
- Correlate beams/flags to note groups
- Assign rhythm types to note groups and **update the note grouping** when neccessary.

Brief summary of these steps are illustrated as below:

<p align='center'>
    <img width="80%" src="figures/rhythm.png">
</p>

The first step is, as mentioned before, to distill beams/flags from all the symbols predicted
by model one. By subtracting with the second model's output, and apply some simple filtering rules,
we get the top-left figure.

Next, the algorithm picks the regions that overlap with known noteheads and stems. We also
get an initial relation between note groups and beams/flags. Both information are kept for
later usage. As a result, the algorithm generates the top-right figure.

The third step is to refine the relation between note groups and beams. Since 
there could be stem of one note group that doesn't overlap with the beam above/below it, and
thus not being included in the same bounding box.  Here, bounding box includes both note group and
beams/flags. This can be adjusted by further scans the region under the bounding box, check
if there contains unknown note groups, and update the relation. Figure is shown in bottom-left.

Finally, the algorithm has all neccessary information to conclude the rhythm types for
each note group now. The algorithm scans a small region for counting how many beams/flags there are.
The region is bounded by the center of the x-axis of the note group, with extention to both left and
right side; the y-axis by the bounding box and the boundary of the note in the note group that
closest to the beams (depending on the direction of the stem). Figure on the bottom-right shows
the region of bounding boxes (green), the scanning range (blue), and the final number of beams/flags
detected by the algorithm. Numeber of rules are also applied to refine the counting result.

In the last step, there is another important mission is to **update the note grouping**, which
means further check the legitmacy of each note group, and separate them into upper and lower
part if neccessary. Since `oemer` takes multi-melody line into consideration, it is not
possible until we collect all the fundamental information to finally determine there is indeed multiple
melody lines in the note group. That is why in the last step here, the algorithm
checks the grouping again.

### Build MusicXML

The process of building MusicXML document follows the **event-based** (objective used in `oemer`
is 'action') mechanism, which essentially means there are different event types, and each
has their own attributes and differently behaviors when being triggered.
The process goes to construct a sequence of events first, and trigger them one-by-one later.
This eventually yields a series of XML strings. A global context is shared across each events,
which plays a key role for holding the music context while decoding.

A brief summary of steps are listed:

1. Sort symbols first by groups, then x-axis position.
2. Initialize the first measure with clef and key information.
3. Determine the alignment between notes/rests in different tracks.
4. Adjust the rhythm of notes/rests or adding rests to make sure the aligned symbols are at the same beat position.
5. Decode the sequence and generate the MusicXML document.

#### Sort

Sort all the instances previously generated by their groups and x-axis, then cluster them into measures.
It's obvious this step is to mitigate how human interpret a music sheet. The status of accidentals are
reset for each measure, rhythm types, chord prgression, etc.


#### Initialize

The initial state of clef type for each track and the key type.
This step includes an important algorithm: key finding. The algorithm can be split down into few steps:

1. Decide if the current measure contains key.

    Check the first few occurance of symbols that are instance of `Sfn`. If there isn't any, return key
    type of C-major.
    If yes, then go to the next step.

2. Define the scan range.

    If the current measure is at the beginning of that row (track), then the first *track_nums* of symbols
    types should be `Clef`, then comes the key.
    Then the end of the scanning, since there are at most 6 sharps/flats of the key (ignoring some special
    cases that the key changes after the double barlines, which may contain naturals), this offset plus
    4 as the tolerance are added to the beginning index.

3. Count occurance

    Count number of occurance of predicted `Sfn` types. Store this information for later process.

4. Check the validity

    Checks if all tracks have the same label (i.e. all flats, all sharps).
    If not, count the most occurance of `Sfn` types. Use this as the label type (i.e. sharp or flat).
    There are more advanced rules being applied in this process. Please check the source code for
    the details.

5. Return key type

    Count the occurance of `Sfn` instances, use the sharp/flat information, and combine the two
    to determine the final key type.


#### Symbol Alignment

Determine the alignment between different notes in different tracks. Notes being paired together (horizontally) are considered at the same beat position in that measure. In other words, notes within the same beat should have the same accumulated beats beforehand across parts. We thus can further use this assumption to adjust the rhythm type of the previous notes.


#### Beat Adjustment

Below shows a graph of alignment results. The number means the minimum detected beat length of the track in that beat position. The commented numbers after each row (beat position) are accumulated length difference.

``` python
# Min durations of each position of the measure

#  Tracks   Accum. Diff.
#  T1  T2
[[ 8., 24.],  # 16
 [ 8.,  0.],  # 8
 [ 8.,  0.],  # 0
 [ 0., 24.],  # 24 <- need to insert an eighth rest to balance the rhythm
 [ 8.,  0.],  # 16                    ↑
 [ 8.,  0.],  # 8                     │ find that
 [ 4.,  4.]]  # 0  <- checkpoint No.2 ┘
```

Checkpoints occur at the row which both have number (meaning both tracks have notes). In the given example, the checkpoints will occur at row 1 and 7. Also, there be a 'mark' to indicate the rhythm in that beat position should be adjusted. The mark will point to where both tracks have number, or the next row after the accumulated difference becomes zero. In above case, the mark will point to row 1, then row 4, then row 7.

At the checkpoint (both have notes), the accumulated difference should be zero. This can be inferred easily from our assumption described in the first paragraph. If the difference is not zero, then the makred position by the 'mark' should adjust their rhythm type or adding rests to make sure the difference go down to zero. Therefore, according to the rule, the total beats in a measure will only increase, since the accumulated difference is always positive number and thus we can only 'add' beat to balance the system.


#### Decode
