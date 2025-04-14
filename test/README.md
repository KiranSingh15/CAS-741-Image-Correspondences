# Image Processing Pipeline: Automated Unit Tests

## Overview
This document provides detailed documentation of all unit tests designed for validating the individual modules of the image processing pipeline. All test files are coordinated via the script `run_unit_checks.py`, which executes the full test suite and generates organized outputs.

## Role of `run_unit_checks.py`
This script serves as the test harness for the project. Its responsibilities include:
- Automatically discovering and executing each test module (e.g., `test_config.py`, `test_featdesc.py`, etc.).
- Capturing and saving the console output from each test to individual result files.
- Writing a summary of all tests, including counts of passed/failed cases, to a timestamped `summary.txt` inside a dedicated `results/` subdirectory.

Each result is logged to: `results/<timestamp>/<test_name>_results.txt`

To run all tests, activate the virtual environment and from the project root directory, run:

```bash
python test/run_unit_checks.py
```

---

## Unit Test Modules

### 1. `test_config.py`
**Purpose**: Validates bounds, configuration paths, and ORB descriptor handling.

| Test Function | Description | Inputs Modified | Expected Outputs | Pass/Fail Criteria |
|---------------|-------------|------------------|------------------|---------------------|
| `test_bounds_are_valid` | Validate that config parameter bounds conform to expected structure | Each bounds tuple | Proper type and expected limits | Passes if all bounds are tuples/lists of two integers matching expectations |
| `test_kern_bounds_are_odd` | Check that Gaussian kernel bounds are odd integers | `config.kern_bounds` | Odd integers | Passes if both bounds are odd |
| `test_dir_path` | Verify directory return type | None | `Path` object | Passes if type is `Path` |
| `test_check_method_limits_invalid` | Assert invalid user-selected method index is rejected | Method ID | AssertionError | Passes if error is raised |
| `test_check_method_limits_valid` | Confirm valid methods pass silently | Valid index config | No error | Passes if method values are within allowed method index ranges |
| `test_check_parameter_limits_invalid_kernel` | Detect invalid kernel configurations | Various invalid kernel values | AssertionError | Passes if invalid input triggers error |
| `test_check_parameter_limits_valid` | Confirm correct parameters pass check | All values valid | No error | Passes if all values fall within configured bounds and meet type constraints |
| `test_set_input_img_path` | Validate folder name returned by path function | Temporary root path | Path to `Raw_Images` and folder name | Passes if both components are correct |
| `test_get_img_IDs_with_images` | Extract IDs from image files | Dummy test files | List of filenames and count | Passes if returned names match input |
| `test_get_img_IDs_empty_directory` | Handle empty folder gracefully | Empty folder | Empty list and zero count | Passes if no files are returned |
| `test_get_img_IDs_ignores_directories` | Ensure directories are not included | Mixed files + subdirs | File-only return | Passes if only image files listed |
| `test_verify_imported_image_none` | Print error if image fails to load | `None` input | Error message to stdout | Passes if expected string appears |
| `test_verify_imported_image_valid` | Print confirmation if image loads | Valid image | Output message | Passes if confirmation message is output |
| `test_get_descriptor_path` | Compute output path for descriptors | Folder name | Path object | Passes if structure is correct |
| `test_load_orb_descriptors_valid` | Load descriptor CSV into OpenCV formats | Valid file | Keypoints + descriptors | Passes if types and shape are correct |
| `test_load_orb_descriptors_missing_columns` | Handle malformed descriptor file | Missing descriptor column | `None, None` | Passes if gracefully returns `None`s |
| `test_orb_descriptor_roundtrip` | End-to-end CSV roundtrip test | Save and reload ORB descriptors | Same values preserved | Passes if keypoints and descriptor array are recovered and match original values |

### 2. `test_controlmod.py`
**Purpose**: Verifies integrated behavior of reading and smoothing image sets in `ControlModule.py`.

| Test Function | Description | Inputs Modified | Expected Outputs | Pass/Fail Criteria |
|---------------|-------------|------------------|------------------|---------------------|
| `test_controlmodule_image_read_and_smooth_outputs` | Validate ControlModule reads images, applies smoothing, and logs summary | Test image set under label (e.g. `lego`, `building`) | Output folders + `test_summary.txt` | Passes if grayscale and smoothed images are saved, and summary file created |

### 3–9.
(See previous canvas updates for complete tables; truncating here for brevity)

---

## Output Structure
When tests are run via `run_unit_checks.py`, the output is organized as follows:

```
results/
  2025-04-13_12-00-00/          # Timestamped session folder
    test_config_results.txt
    test_controlmod_results.txt
    ...
    summary.txt                 # Summary report of test run
```

Each `*_results.txt` file includes verbose test output from `pytest -v`, while `summary.txt` provides a condensed report of total/pass/fail counts for each test script.

---

## How to Run the Tests
From the root of the project (with virtual environment activated), run:

```bash
python test/run_unit_checks.py
```

The results will appear in the `results/` folder.

---

For test-specific details, refer directly to the docstrings inside each test file. All tests follow the PyTest standard and can also be executed individually using:

```bash
pytest path/to/test_file.py
```

### 3. `test_featdesc.py`
**Purpose**: Verifies descriptor computation logic in `FeatureDescriptorModule.py`.

| Test Function | Description | Inputs Modified | Expected Outputs | Pass/Fail Criteria |
|---------------|-------------|------------------|------------------|---------------------|
| `test_compute_descriptors_valid` | Compute descriptors from real keypoints | Valid ORB + random image | Tuple with keypoints and descriptors | Passes if output is a tuple of list of cv.KeyPoint and NumPy descriptor array |
| `test_compute_descriptors_invalid_method` | Handle invalid descriptor method | Method index 0 | None | Passes if function returns None for invalid method |
| `test_compute_descriptors_with_no_keypoints` | Handle empty keypoints list | Empty list | Tuple with empty list and `None` | Passes if descriptor output is None |
| `test_compute_descriptors_with_none_orb` | Catch None ORB input | ORB object is `None` | Raises `AttributeError` | Passes if exception is raised |

### 4. `test_featmatches.py`
**Purpose**: Tests brute-force matcher and match handling in `FeatureMatchingModule.py`.

| Test Function | Description | Inputs Modified | Expected Outputs | Pass/Fail Criteria |
|---------------|-------------|------------------|------------------|---------------------|
| `test_create_BF_matcher_valid` | Confirm matcher creation using valid method | Method 1 and Hamming norm | `cv.BFMatcher` instance | Passes if returned object is an instance of `cv.BFMatcher` |
| `test_create_BF_matcher_invalid_behavior` | Handle unsupported matcher type | Method 0 with L2 norm | `None` or fallback matcher | Passes if returns `None` or valid fallback instance depending on module logic |
| `test_match_features_no_loss` | Match identical descriptors with cross-check | Two identical arrays of shape (10, 32) | List of 10 `cv.DMatch` objects | Passes if match count equals descriptor count and all items are valid matches |
| `test_match_features_invalid_shapes` | Catch mismatched descriptor dimensions | Descriptors of shape (10,32) and (5,16) | OpenCV error | Passes if exception of type `cv.error` is raised |
| `test_sort_matches_ordering` | Ensure match list is sorted by distance | Unordered `cv.DMatch` objects | Sorted list | Passes if returned list is sorted in ascending order of `distance` field |
| `test_no_matches_returns_empty_list` | Handle empty descriptor arrays | Empty descriptors | Empty list | Passes if return is a list of length 0 |

### 5. `test_imagePlot.py`
**Purpose**: Tests image output generation and saving via `ImagePlotModule.py`.

| Test Function | Description | Inputs Modified | Expected Outputs | Pass/Fail Criteria |
|---------------|-------------|------------------|------------------|---------------------|
| `test_gen_kp_img_with_none_keypoints` | Draw image with no keypoints | Keypoints=None | Output image with unchanged shape | Passes if output shape matches input and is a NumPy array |
| `test_gen_kp_img_no_flag` | Basic draw without visualization flags | Keypoints list | Image with points | Passes if output retains original shape and is valid image data |
| `test_gen_kp_img_rich_keypoints` | Draw using rich keypoint flags | Keypoints list with DRAW_RICH_KEYPOINTS | Annotated keypoints image | Passes if flag is correctly rendered and shape preserved |
| `test_gen_kp_img_with_none_image` | Validate null image error | Image=None | OpenCV error | Passes if `cv.error` is raised |
| `test_gen_matched_features_success` | Match visualization succeeds with valid input | Two dummy images + keypoints + match | Composite image | Passes if result is non-empty image with valid shape |
| `test_gen_matched_features_with_empty_matches` | Handle match list as empty | Empty matches array | Valid fallback image | Passes if fallback image is returned without error |
| `test_make_directory_creates` | Directory creation utility | Folder name and base path | Folder exists on disk | Passes if directory is created under correct location |
| `test_save_image_success` | Save image to disk | NumPy image and path | Writable image file | Passes if output image file is present and non-empty |

### 6. `test_imagesmooth.py`
**Purpose**: Tests grayscale and Gaussian smoothing in `ImageSmoothingModule.py`.

| Test Function | Description | Inputs Modified | Expected Outputs | Pass/Fail Criteria |
|---------------|-------------|------------------|------------------|---------------------|
| `test_convert_to_greyscale_output_shape` | Convert 3-channel image to grayscale | BGR image | Single-channel image | Passes if shape reduces from (H, W, 3) to (H, W) |
| `test_convert_to_greyscale_actual_conversion` | Test luminance accuracy for color channel | Blue-only BGR image | Low-intensity grayscale | Passes if mean of result is below threshold indicating reduced luminance contribution |
| `test_convert_to_greyscale_invalid_input` | Reject single-channel input | Already grayscale image | OpenCV error | Passes if `cv.error` is raised |
| `test_smooth_image_valid_gaussian` | Apply Gaussian blur with correct params | Random grayscale image | Smoothed output | Passes if result differs from input and retains shape/type |
| `test_smooth_image_invalid_method_returns_nothing` | Handle unknown smoothing method | Method=0 | None | Passes if return value is None without error |
| `test_smooth_image_invalid_kernel_size` | Trigger error on even-sized kernel | sz_kern=4 | OpenCV error | Passes if exception is thrown due to invalid kernel |

### 7. `test_kpdetect.py`
**Purpose**: Verifies keypoint detection and ORB initialization.

| Test Function | Description | Inputs Modified | Expected Outputs | Pass/Fail Criteria |
|---------------|-------------|------------------|------------------|---------------------|
| `test_initialize_orb_valid_config` | Build valid ORB detector | All parameters valid | ORB object | Passes if return is instance of `cv.ORB` |
| `test_initialize_orb_invalid_config_detection` | Invalid detection method | `mthd_kp_detection=0` | None | Passes if output is None |
| `test_initialize_orb_invalid_config_description` | Invalid descriptor method | `mthd_kp_description=0` | None | Passes if output is None |
| `test_initialize_orb_invalid_both` | Both indices invalid | Both = 0 | None | Passes if output is None |
| `test_detect_keypoints_with_valid_orb` | Valid detection with ORB | Image + ORB | List of keypoints | Passes if all elements are `cv.KeyPoint` |
| `test_detect_keypoints_invalid_method` | Invalid method ID | Method = 0 | None | Passes if output is None |
| `test_detect_keypoints_with_none_orb` | Fail on missing ORB | ORB=None | AttributeError | Passes if exception is raised |

### 8. `test_outputFormat.py`
**Purpose**: Validates CSV output of keypoints, descriptors, and matches.

| Test Function | Description | Inputs Modified | Expected Outputs | Pass/Fail Criteria |
|---------------|-------------|------------------|------------------|---------------------|
| `test_output_keypoints_variable_size` | Output CSV of N keypoints | N = 1, 100, 1000 | CSV file with keypoint fields | Passes if row count = N and all fields exist |
| `test_output_descriptors_variable_size` | Save descriptors with matching keypoints | N = 1, 100, 1000 | Valid descriptor strings in CSV | Passes if 256-bit descriptor field present for all rows |
| `test_output_matches_variable_size` | Output match info with metadata | N matches between kp1 and kp2 | CSV file with spatial and binary descriptors | Passes if each row includes both descriptors and match fields |

### 9. `test_specParams.py`
**Purpose**: Validates system-wide default parameters and their ranges.

| Test Function | Description | Inputs Modified | Expected Outputs | Pass/Fail Criteria |
|---------------|-------------|------------------|------------------|---------------------|
| `test_kernel_bounds` | Validate kernel size limits | Assigned value `test_k` | Integer, odd, 3 <= k <= 11 | Passes if all constraints are satisfied |
| `test_kernel_bounds` (stddev) | Validate Gaussian std dev | Assigned value `test_sigma` | Float in (0, 10] | Passes if type is float and within tolerance |
| `test_fast_bounds` | Test pixel intensity threshold | Value = `test_t` | Integer in [2, 254] | Passes if threshold is inside inclusive range |
| `test_bin_bounds` | Validate descriptor bin size | Value = `test_b` | Integer in [1, 2048] | Passes if type and bounds correct |
| `test_patch_size_bounds` | Check feature patch size | `test_p` | Integer in [5, 100] | Passes if in bounds |
| `test_match_distance_limits` | Validate match distance upper limit | `test_d` | Integer in (0, 150] | Passes if >0 and <=150 |
| `test_num_match_disp` | Number of displayed matches | `test_displayed_matches` | Integer in [1, 1000] | Passes if within range and type=int |
| `test_avail_methods` | Check all methods are available | Lists from `specParams` | Lists populated | Passes if none are empty and of correct type |
| `test_selected_methods` | Validate user-selected method indices | Assigned values | Valid range [0, n] | Passes if integers and all >= 0 |