import * as React from 'react'
import DefaultDraggable from './Draggable'

export interface DraggableWrapperProps {
  children?: any
  onMouseDown?: any
  onTouchStart?: any
}

const defaultStyle: React.CSSProperties = {
  position: 'absolute',
  top: 0,
  left: 0,
  touchAction: 'none',
}

const DraggableWrapper: React.StatelessComponent<DraggableWrapperProps> = ({
  children: Draggable = <DefaultDraggable size={40} />,
  onMouseDown,
  onTouchStart,
}) => {
  return (
    <div style={defaultStyle} onMouseDown={onMouseDown} onTouchStart={onTouchStart}>
      {Draggable}
    </div>
  )
}

export default DraggableWrapper
