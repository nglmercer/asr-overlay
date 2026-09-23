// Lets native `tsc` typecheck imports of SFCs (as untyped components)
// until vue-tsc supports TypeScript 7.
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}
