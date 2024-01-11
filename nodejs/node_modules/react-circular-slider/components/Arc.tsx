import * as React from 'react'
import { arc } from 'd3-shape'

export interface ArcProps {
  startAngle?: number
  endAngle?: number
  padding: number
  radius: number
  thickness: number
  style?: React.CSSProperties
  mask?: JSX.Element
  [index: string]: any
}

const defaultStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
}

export default ({
  radius = 80,
  startAngle = 0,
  endAngle = 360,
  padding = 10,
  thickness = 20,
  style,
  children,
  mask,
  ...otherProps,
}: ArcProps): JSX.Element => {
  const arcData = arc()({
    innerRadius: radius - thickness,
    outerRadius: radius,
    startAngle: Math.PI * 2 * (startAngle / 360),
    endAngle: Math.PI * 2 * (endAngle / 360),
    padAngle: 0,
  })

  const size = (padding + radius) * 2
  return (
    <svg
      height={size}
      width={size}
      viewBox={`0 0 ${size} ${size}`}
      style={{ ...defaultStyle, ...style }}
      {...otherProps}
    >
      {mask && (
        <defs>
          <mask id="arcmask">
            <rect width={size} height={size} fill="white" />
            {mask}
          </mask>
        </defs>
      )}
      <g mask="url(#arcmask)">
        <path d={arcData!} transform={`translate(${size / 2} ${size / 2})`} />
        {children}
      </g>
    </svg>
  )
}
