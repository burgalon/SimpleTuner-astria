<template>
    <div class="sidebar">
      <h2>Soft Regions</h2>
      <div
        v-for="sr in regionData.softRegions"
        :key="sr.id"
        class="region-item"
      >
        <label>Soft Region {{sr.id}}:</label>
        <textarea v-model="sr.prompt"></textarea>
      </div>
  
      <h2>Hard Regions</h2>
      <div
        v-for="hb in regionData.hardRegions"
        :key="hb.id"
        class="region-item"
      >
        <label>Hard Region {{hb.id}}:</label>
        <textarea v-model="hb.prompt"></textarea>
      </div>
    </div>
  </template>
  
  <script setup>
  import { toRefs } from 'vue'
  
  const props = defineProps({
    regionData: {
      type: Object,
      required: true
    }
  })
  
  /**
   * Because regionData is passed in from App.vue as a reactive object,
   * we can directly modify it here (v-model on textareas).
   * This will keep everything in sync with the parent and other siblings.
   */
  </script>
  
  <style scoped>
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .region-item {
    margin-bottom: 1rem;
  }
  textarea {
    width: 100%;
    height: 4em;
  }
  </style>