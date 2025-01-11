<template>
  <v-stage
    ref="stageRef"
    :width="stageWidth"
    :height="stageHeight"
  >
    <!-- Soft Region Layer -->
    <v-layer>
      <template v-for="sr in regionData.softRegions" :key="sr.id">
        <v-rect
          :x="sr.x"
          :y="sr.y"
          :width="sr.width"
          :height="sr.height"
          stroke="green"
          stroke-width="2"
          :draggable="false"
          @dragmove="onSoftRegionDrag($event, sr)"
          @transformend="onSoftRegionTransform($event, sr)"
        />
        <!-- Possibly show text label or text editor on top -->
        <v-text
          :x="sr.x + 5"
          :y="sr.y + 5"
          :text="sr.prompt"
          font-size="14"
          fill="green"
        />
      </template>
    </v-layer>

    <!-- Hard Region Layer -->
    <v-layer>
      <template v-for="hb in regionData.hardRegions" :key="hb.id">
        <v-rect
          :x="hb.x"
          :y="hb.y"
          :width="hb.width"
          :height="hb.height"
          stroke="red"
          stroke-width="2"
          :draggable="true"
          @dragmove="onHardRegionDrag($event, hb)"
          @transformend="onHardRegionTransform($event, hb)"
          @mousedown="handleStageMouseDown"
          @touchstart="handleStageMouseDown"
        />
        <v-text
          :x="hb.x + 5"
          :y="hb.y + 5"
          :text="hb.prompt"
          font-size="14"
          fill="red"
        />
      </template>
    </v-layer>
    <v-layer>
      <v-transformer ref="transformer" />
    </v-layer>
  </v-stage>
</template>

<script>
import { onMounted, ref } from 'vue'

export default {
  name: 'CanvasEditor',
  props: {
    // We no longer require 'initialJson'. Instead, we receive the parent's regionData directly.
    regionData: {
      type: Object,
      required: true
    },
    stageWidth: {
      type: Number,
      default: 1024
    },
    stageHeight: {
      type: Number,
      default: 1024
    }
  },
  setup(props) {
    const stageRef = ref(null)

    // The parent has already loaded or will load 'regionData' from JSON.
    // We do NOT create our own local copy. We directly mutate props.regionData.

    function onSoftRegionDrag(evt, sr) {
      const shape = evt.target
      sr.x = shape.x()
      sr.y = shape.y()
    }

    function onSoftRegionTransform(evt, sr) {
      const shape = evt.target
      sr.x = shape.x()
      sr.y = shape.y()
      sr.width = shape.width() * shape.scaleX()
      sr.height = shape.height() * shape.scaleY()
      // Reset Konva’s scale to 1 after the transform
      shape.scaleX(1)
      shape.scaleY(1)
    }

    function onHardRegionDrag(evt, hb) {
      const shape = evt.target
      hb.x = shape.x()
      hb.y = shape.y()
      // Optionally enforce that it must remain within its parent SR’s boundaries
    }

    function onHardRegionTransform(evt, hb) {
      const shape = evt.target
      hb.x = shape.x()
      hb.y = shape.y()
      hb.width = shape.width() * shape.scaleX()
      hb.height = shape.height() * shape.scaleY()
      shape.scaleX(1)
      shape.scaleY(1)
    }

    onMounted(() => {
      const stage = stageRef.value.getStage()
      const layer = stage.getLayers()[0] // Assuming hard regions are on the second layer

      props.regionData.hardRegions.forEach((hb, index) => {
        const rect = layer.findOne(`#hardRegion-${index}`)
        const transformer = new Konva.Transformer({
          nodes: [rect],
          boundBoxFunc: (oldBox, newBox) => {
            const softRegion = props.regionData.softRegions.find(sr => sr.id === hb.parentId)
            if (!softRegion) return oldBox

            const maxX = softRegion.x + softRegion.width - newBox.width
            const maxY = softRegion.y + softRegion.height - newBox.height

            if (newBox.x < softRegion.x || newBox.y < softRegion.y || newBox.x > maxX || newBox.y > maxY) {
              return oldBox
            }
            return newBox
          }
        })
        layer.add(transformer)
      })
    })

    return {
      stageRef,
      onSoftRegionDrag,
      onSoftRegionTransform,
      onHardRegionDrag,
      onHardRegionTransform
    }
  },

  methods: {
    handleTransformEnd(e) {
      // shape is transformed, let us save new attrs back to the node
      // find element in our state
      const rect = this.rectangles.find(
        (r) => r.name === this.selectedShapeName
      );
      // update the state
      rect.x = e.target.x();
      rect.y = e.target.y();
      rect.rotation = e.target.rotation();
      rect.scaleX = e.target.scaleX();
      rect.scaleY = e.target.scaleY();

      // change fill
      rect.fill = Konva.Util.getRandomColor();
    },

    handleStageMouseDown(e) {
      // clicked on stage - clear selection
      if (e.target === e.target.getStage()) {
        this.selectedShapeName = '';
        this.updateTransformer();
        return;
      }

      // clicked on transformer - do nothing
      const clickedOnTransformer =
        e.target.getParent().className === 'Transformer';
      if (clickedOnTransformer) {
        return;
      }

      // find clicked rect by its name
      console.log('e.target', e.target);
      const name = e.target.name();
      console.log('name', name);
      const rect = this.regionData.hardRegions.find((r) => r.name === name);
      if (rect) {
        this.selectedShapeName = name;
      } else {
        this.selectedShapeName = '';
      }
      this.updateTransformer();
    },

    updateTransformer() {
      debugger;
      // here we need to manually attach or detach Transformer node
      const transformerNode = this.$refs.transformer.getNode();
      const stage = transformerNode.getStage();
      const { selectedShapeName } = this;

      const selectedNode = stage.findOne('.' + selectedShapeName);
      // do nothing if selected node is already attached
      if (selectedNode === transformerNode.node()) {
        return;
      }

      if (selectedNode) {
        // attach to another node
        transformerNode.nodes([selectedNode]);
      } else {
        // remove transformer
        transformerNode.nodes([]);
      }
    },
  },
}
</script>

<style scoped>
/* Example styling */
</style>