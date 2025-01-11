<template>
  <div class="app-container">
    <!-- Sidebar for editing region text/prompts -->
    <sidebar-editor
      :regionData="regionData"
      style="flex: 1; border-right: 1px solid #ccc; padding: 1rem;">
    </sidebar-editor>

    <!-- Main Canvas Editor -->
    <canvas-editor
      :regionData="regionData"
      :stageWidth="stageWidth"
      :stageHeight="stageHeight"
      style="flex: 2;">
    </canvas-editor>

    <!-- Example: a button to export or save the data -->
    <div class="export-panel">
      <button @click="exportJson">Export to JSON</button>
      <pre>{{ exportedJson }}</pre>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import CanvasEditor from './components/CanvasEditor.vue'
import SidebarEditor from './components//SidebarEditor.vue'

// This could be your initial JSON. 
// Or fetch from server, or import from an external file
import initialJson from './assets/sampleData.json'

// Provide default stage size
const stageWidth = 1024
const stageHeight = 1024

// regionData is shared between the sidebar and canvas
// We'll store all the SoftRegions and HardRegions here
const regionData = reactive({
  softRegions: [],
  hardRegions: []
})

// For demonstration: a string of the "exported" result
const exportedJson = ref('')

// On mount, parse or transform the JSON into pixel-based data
onMounted(() => {
  loadFromJson(initialJson)
})

/**
 * parseRatioString("0.3,1;0.5,0.33,0.34,0.33;0.2,1")
 * => [ [0.3, 1], [0.5, 0.33, 0.34, 0.33], [0.2, 1] ]
 * Each sub-array represents: [ rowHeightRatio, col1Ratio, col2Ratio, ... ]
 */
 function parseRatioString(ratioString) {
  if (!ratioString.trim()) return []
  const rowsStr = ratioString.split(';')
  return rowsStr.map((rowStr) => {
    // split by comma, filter out empty strings
    const cols = rowStr
      .split(',')
      .map((x) => x.trim())
      .filter(Boolean)
    return cols.map(parseFloat)
  })
}

/**
 * cumsum([0.3, 0.5, 0.2]) => [0.3, 0.8, 1.0]
 */
function cumsum(values) {
  let running = 0
  return values.map((v) => {
    running += v
    return running
  })
}


function loadFromJson(json) {
  // 1) Clear out any existing data
  regionData.softRegions = []
  regionData.hardRegions = []

  // 2) Split the SR_prompt by "BREAK" → array of sub-prompts
  const srPrompts = json.SR_prompt
    ? json.SR_prompt.split("BREAK").map((p) => p.trim())
    : []

  // 3) Parse SR_hw_split_ratio into row-based arrays
  //    e.g. "0.3,1;0.5,0.33,0.34,0.33;0.2,1"
  const ratioString = json.SR_hw_split_ratio || ""
  const rows = parseRatioString(ratioString)
  // rows might look like: [
  //   [0.3, 1],
  //   [0.5, 0.33, 0.34, 0.33],
  //   [0.2, 1]
  // ]

  // 4) Compute row boundaries in fraction coords
  //    We'll assume row[0] is the row-height ratio
  const rowHeights = rows.map((r) => r[0])           // e.g. [0.3, 0.5, 0.2]
  const rowBoundaries = [0, ...cumsum(rowHeights)]   // e.g. [0, 0.3, 0.8, 1.0]

  // 5) Build Soft Regions row by row
  let srIdCounter = 1
  let srPromptIndex = 0

  rows.forEach((row, rowIdx) => {
    const topFrac = rowBoundaries[rowIdx]
    const botFrac = rowBoundaries[rowIdx + 1]

    // The remaining values after row[0] are the column splits
    // e.g. [0.33, 0.34, 0.33] that sum to 1 (in principle).
    const colRatios = row.slice(1)
    // If empty, assume a single column occupying full width
    const colRatiosOrDefault = colRatios.length ? colRatios : [1.0]
    const colBoundaries = [0, ...cumsum(colRatiosOrDefault)]

    for (let c = 0; c < colBoundaries.length - 1; c++) {
      // Convert frac → pixel
      const leftFrac = colBoundaries[c]
      const rightFrac = colBoundaries[c + 1]
      const x = leftFrac * stageWidth
      const y = topFrac * stageHeight
      const w = (rightFrac - leftFrac) * stageWidth
      const h = (botFrac - topFrac) * stageHeight

      // Soft region prompt (if we exceed srPrompts, fallback to empty)
      const prompt = srPrompts[srPromptIndex] || null
      if (prompt === null) break
      srPromptIndex++

      regionData.softRegions.push({
        id: srIdCounter++,
        x,
        y,
        width: w,
        height: h,
        prompt
      })
    }
  })

  // 6) Parse Hard Regions from fraction-based arrays
  //    E.g. HB_prompt_list, HB_m_offset_list, etc.
  const hbPrompts = json.HB_prompt_list || []
  const mOffsets = json.HB_m_offset_list || []
  const nOffsets = json.HB_n_offset_list || []
  const mScales = json.HB_m_scale_list || []
  const nScales = json.HB_n_scale_list || []

  // We'll assume all these arrays have the same length
  const countHB = hbPrompts.length
  let hbIdCounter = 100

  for (let i = 0; i < countHB; i++) {
    const xOffset = mOffsets[i] * stageWidth
    const yOffset = nOffsets[i] * stageHeight
    const wScale = mScales[i] * stageWidth
    const hScale = nScales[i] * stageHeight

    // Optionally, we can do advanced logic to find which Soft Region
    // encloses this Hard Region. For now, we'll set parentSoftRegionId = null
    // or pick the region that matches best.
    let parentSoftRegionId = null

    regionData.hardRegions.push({
      id: hbIdCounter++,
      parentSoftRegionId,
      x: xOffset,
      y: yOffset,
      width: wScale,
      height: hScale,
      prompt: hbPrompts[i]
    })
  }
}

function exportJson() {
  const newJson = {
    SR_hw_split_ratio: '',
    SR_prompt: '',
    HB_prompt_list: [],
    HB_m_offset_list: [],
    HB_n_offset_list: [],
    HB_m_scale_list: [],
    HB_n_scale_list: []
  }

  // --- 1) Group Soft Regions into logical rows ---
  // Sort all soft regions by their top edge (y)
  const sortedSR = [...regionData.softRegions].sort((a, b) => a.y - b.y)

  // We define a small threshold to decide if a region belongs in a new row
  // or the existing row. Adjust as needed.
  const ROW_THRESHOLD = 5 // px

  const rowGroups = []
  let currentRow = []
  let currentRowBottom = -1

  for (const sr of sortedSR) {
    const srTop = sr.y
    const srBottom = sr.y + sr.height

    // If this is the first region or if there's a gap bigger than threshold:
    if (!currentRow.length || srTop - currentRowBottom > ROW_THRESHOLD) {
      // Start a new row
      if (currentRow.length) {
        rowGroups.push(currentRow)
      }
      currentRow = [sr]
      currentRowBottom = srBottom
    } else {
      // Same row
      currentRow.push(sr)
      currentRowBottom = Math.max(currentRowBottom, srBottom)
    }
  }
  // Push the last row if it's not empty
  if (currentRow.length) {
    rowGroups.push(currentRow)
  }

  // For each row group, we’ll:
  //  - find the row’s min-y, max-y => row height in px => row height ratio
  //  - sort sub-regions by x => compute each region’s width ratio
  //  - build an array: [rowHeightRatio, col1Ratio, col2Ratio, ...]
  //  - also gather the prompts in that row (left→right)

  const rowRatioArrays = []
  const srPromptList = []

  rowGroups.forEach((row) => {
    // Sort the row by x
    row.sort((a, b) => a.x - b.x)

    // row's minY, maxY
    const minY = Math.min(...row.map((r) => r.y))
    const maxY = Math.max(...row.map((r) => r.y + r.height))
    const rowHeightPx = maxY - minY
    const rowHeightRatio = rowHeightPx / stageHeight

    // compute column ratios in that row
    const colRatios = row.map((r) => (r.width / stageWidth))

    // gather prompts in left→right order
    row.forEach((r) => srPromptList.push(r.prompt))

    // The ratio array: first = rowHeight, then col splits
    // e.g. [0.3, 1], or [0.5, 0.33, 0.34, 0.33], etc.
    const ratioArray = [rowHeightRatio, ...colRatios]
    rowRatioArrays.push(ratioArray)
  })

  // --- 2) Rebuild SR_hw_split_ratio string
  // Turn each ratio array [0.3, 1] into "0.3,1" etc. Then join them with ';'
  const srHwSplitRatio = rowRatioArrays
    .map(rowArr => rowArr.map(num => num.toFixed(4)).join(','))
    .join(';')

  newJson.SR_hw_split_ratio = srHwSplitRatio

  // --- 3) Rebuild SR_prompt by joining prompts with " BREAK "
  // in the same order we collected them above
  const srPromptString = srPromptList.join(' BREAK ')
  newJson.SR_prompt = srPromptString

  // --- 4) Convert Hard Regions to fraction-based offsets
  regionData.hardRegions.forEach((hb) => {
    const mOffset = hb.x / stageWidth
    const nOffset = hb.y / stageHeight
    const mScale = hb.width / stageWidth
    const nScale = hb.height / stageHeight

    newJson.HB_prompt_list.push(hb.prompt)
    newJson.HB_m_offset_list.push(mOffset)
    newJson.HB_n_offset_list.push(nOffset)
    newJson.HB_m_scale_list.push(mScale)
    newJson.HB_n_scale_list.push(nScale)
  })

  // --- 5) Final JSON
  exportedJson.value = JSON.stringify(newJson, null, 2)
}
</script>

<style scoped>
.app-container {
  display: flex;
  height: 80vh;
}

.export-panel {
  margin-top: 1rem;
  padding: 1rem;
  background: #f9f9f9;
}
</style>