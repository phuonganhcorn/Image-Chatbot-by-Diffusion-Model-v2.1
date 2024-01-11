import { scaleLinear } from 'd3-scale'

export interface Point {
  x: number
  y: number
}

export const calculateOrigin = (container: Element, { x: offsetX, y: offsetY }: Point) => {
  const { x: containerX, y: containerY } = absoluteContainerPosition(container)!
  return { x: containerX + offsetX, y: containerY + offsetY }
}

export const calculateAngleDelta = ({ x: x1, y: y1 }: Point, { x: x2, y: y2 }: Point) =>
  Math.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)

export const calculateAngleFromOrigin = ({ x: originX, y: originY }: Point, { x: pointX, y: pointY }: Point) => {
  let angleInRadians = Math.atan2(pointY - originY, pointX - originX)
  if (angleInRadians < 0) {
    angleInRadians += 2 * Math.PI
  }
  return angleInRadians
}

export const calculateAngleToPoint = (container: Element, offset: Point, point: Point) => {
  const origin = calculateOrigin(container, offset)
  return calculateAngleFromOrigin(origin, point)
}

export const calculateAngleBetweenPoints = ({ x: originX, y: originY }: Point, {x: startX, y: startY}: Point, {x: endX, y: endY}: Point) => {
  const startPoint = { x: startX - originX, y: startY - originY }
  const endPoint = { x: endX - originX, y: endY - originY }
  return calculateAngleDelta(startPoint, endPoint)
}

export const calculateRadialPosition = (container: Element, offset: Point, radius: number, point: Point) => {
  const angleInRadians = calculateAngleToPoint(container, offset, point)
  return {
    x: offset.x + radius * Math.cos(angleInRadians),
    y: offset.y + radius * Math.sin(angleInRadians),
  }
}

export const calculateRadialPositionFromValue = (offset: Point, radius: number, value = 0, minValue = 0, maxValue = 100) => {
  const angleInRadians = valueToRadians(value, minValue, maxValue)
  return {
    x: offset.x + radius * Math.cos(angleInRadians * Math.PI),
    y: offset.y + radius * Math.sin(angleInRadians * Math.PI),
  }
}

export const absoluteContainerPosition = (container: Element) => {
  const { left: x, top: y } = container.getBoundingClientRect()
  return { x, y }
}

export const valueToRadians = (value: number, minValue: number, maxValue: number) => scaleLinear().domain([minValue, maxValue]).range([0, 2])(value)

export const angleToValue = (angle: number, minValue: number, maxValue: number) => scaleLinear().domain([0, Math.PI * 2]).range([minValue, maxValue])(angle)
