import * as React from 'react'

export const absoluteTouchPosition = (e: React.TouchEvent<SVGElement>) => ({
  x: e.changedTouches[0].pageX - (window.scrollX || window.pageXOffset),
  y: e.changedTouches[0].pageY - (window.scrollY || window.pageYOffset),
})

export const absoluteMousePosition = (e: React.MouseEvent<SVGElement>) => ({
  x: e.pageX - (window.scrollX || window.pageXOffset),
  y: e.pageY - (window.scrollY || window.pageYOffset),
})
