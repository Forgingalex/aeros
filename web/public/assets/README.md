# AEROS Assets

Place the AEROS logo symbol image here.

## Logo Requirements

- **File name**: `aeros-logo-symbol.png` (or `.svg`)
- **Recommended size**: 40x40px to 80x80px
- **Format**: PNG with transparency or SVG
- **Content**: Abstract geometric flight vector icon (arrow passing through oval)

The logo will be displayed in the dashboard header at 40x40px.

## Current Usage

The logo is referenced in `AerosDashboard.tsx` as:
```tsx
<img
  src="/assets/aeros-logo-symbol.png"
  alt="Aeros logo"
  className="aeros-logo-symbol"
/>
```

If you're using the SVG logo component instead, you can replace this with the `<Logo />` component from `components/Logo.js`.

