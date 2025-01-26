import cv2
import json
import numpy as np

prompt_regions = None
with open('test_regional_three_lora.json', 'r') as f:
    prompt_regions = json.load(f)

SR_hw_split_ratio = prompt_regions['SR_hw_split_ratio']
SR_prompt = prompt_regions['SR_prompt']
HB_prompt_list = prompt_regions['HB_prompt_list']
HB_m_offset_list = prompt_regions['HB_m_offset_list']
HB_n_offset_list = prompt_regions['HB_n_offset_list']
HB_m_scale_list = prompt_regions['HB_m_scale_list']
HB_n_scale_list = prompt_regions['HB_n_scale_list']

if ';' not in SR_hw_split_ratio:
    SR_hw_split_ratio = f"1.0,{SR_hw_split_ratio}"

# --- Create a blank image ---
height, width = 1024, 1024
image = np.ones((height, width, 3), dtype=np.uint8) * 255

# ------------------------------------------------------------------------------
# 1) Draw "HB" bounding boxes (same as before)
# ------------------------------------------------------------------------------
for i, label in enumerate(HB_prompt_list):
    x_off = int(HB_m_offset_list[i] * width)
    y_off = int(HB_n_offset_list[i] * height)
    w = int(HB_m_scale_list[i] * width)
    h = int(HB_n_scale_list[i] * height)

    top_left = (x_off, y_off)
    bottom_right = (x_off + w, y_off + h)

    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
    cv2.putText(
        image, label,
        (x_off, max(y_off - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA
    )

# ------------------------------------------------------------------------------
# 2) Parse and draw “SB” (soft boundary) regions from SR_hw_split_ratio
# ------------------------------------------------------------------------------
def parse_ratio_string(ratio_string):
    """
    Splits a semicolon-delimited string (rows) into lists of floats.
    Each row is further split by comma into columns of floats.
    Example: "0.3,1;0.7,0.2,0.6,0.2"
       => rows = [[0.3, 1], [0.7, 0.2, 0.6, 0.2]]
    """
    rows_str = ratio_string.split(';')
    rows_float = []
    for row_str in rows_str:
        cols = [float(x) for x in row_str.split(',') if x.strip() != ""]
        rows_float.append(cols)
    return rows_float

def cumsum(values):
    """Simple cumulative sum of a list of floats."""
    out = []
    running = 0.0
    for v in values:
        running += v
        out.append(running)
    return out

# Split the SR_prompt by "BREAK" to get sub-prompts (soft boundary text).
sb_prompts = [p.strip() for p in SR_prompt.split("BREAK")]
for sb_prompt in sb_prompts:
    print(sb_prompt)
# The total number of SB cells must match the total sub-prompts.  
# If not, you may need to adapt the indexing or skip extra sub-prompts.

# Parse the SR_hw_split_ratio into rows of floats
rows = parse_ratio_string(SR_hw_split_ratio)
# Example:
#   rows[0] = [0.3, 1]           (2 “columns” in row 1)
#   rows[1] = [0.7, 0.2, 0.6, 0.2]  (4 “columns” in row 2)
#
# Typically, the first float of each row is that row’s “height ratio” 
# and the remaining floats in that row sum to 1 for the columns (or vice versa),
# but it depends on how you intend to structure the ratio. 
# For demonstration, let's assume the first number is the row height ratio, 
# and the rest are column splits (summing to 1).  

# Step A: separate out the row-height-ratios from each row’s column-splits
row_heights = []
row_cols = []
for row in rows:
    # The first value is the row height ratio
    row_heights.append(row[0])
    # The rest are column width ratios
    # (some rows might have only 1 column => row[1:] = [1] if you want full width)
    row_cols.append(row[1:])

# Step B: get cumulative sums for row heights
row_boundaries = [0.0] + cumsum(row_heights)  # e.g. [0, 0.3, 1.0]
# row_boundaries[i], row_boundaries[i+1] => top & bottom of row i in fractional coords

# For each row, get cumulative sums for its columns
all_col_boundaries = []
for cols in row_cols:
    if len(cols) == 0:
        # If no columns, default to a single column occupying full width
        all_col_boundaries.append([0.0, 1.0])
    else:
        col_b = [0.0] + cumsum(cols)
        all_col_boundaries.append(col_b)

# Now we have row boundaries in fractional coords and column boundaries for each row.
# We can iterate through each row and each column cell, then draw a rectangle.

sb_index = 0  # index for sub-prompts
for row_idx in range(len(row_heights)):
    top_frac = row_boundaries[row_idx]
    bot_frac = row_boundaries[row_idx + 1]

    # This row’s set of column boundaries
    col_boundaries = all_col_boundaries[row_idx]
    num_cols = len(col_boundaries) - 1

    for c in range(num_cols):
        if sb_index >= len(sb_prompts):
            break  # if more cells than sub-prompts, stop or handle however you wish

        left_frac = col_boundaries[c]
        right_frac = col_boundaries[c + 1]

        # Convert fractional coords to pixel coords
        top_pix = int(top_frac * height)
        bot_pix = int(bot_frac * height)
        left_pix = int(left_frac * width)
        right_pix = int(right_frac * width)

        # Draw a green rectangle for the “SB” region
        cv2.rectangle(
            image,
            (left_pix, top_pix),
            (right_pix, bot_pix),
            (0, 180, 0),  # green color
            2
        )

        # Label the rectangle with the corresponding sub-prompt
        # (placing text near the top-left corner of that cell)
        cv2.putText(
            image,
            f"SB: {sb_prompts[sb_index]}",
            (left_pix + 5, top_pix + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 180, 0),
            1,
            cv2.LINE_AA
        )
        sb_index += 1

# --- Finally, display the combined result ---
cv2.imwrite('regions.png', image)
