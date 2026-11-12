-- Pandoc understands the paper's LaTeX; these blocks select IOS Word styles.
---@param element pandoc.Div
---@return pandoc.Div
function Div(element)
  for _, class in ipairs(element.classes) do
    if class:sub(1, 3) == 'ios' then
      element.attributes['custom-style'] = class:sub(4)
    end
  end
  return element
end

---@type integer[]
local sections = {0, 0, 0, 0}

---@param element pandoc.Header
---@return pandoc.Div
function Header(element)
  local style = 'HeadingUnn' .. element.level
  local title = pandoc.utils.stringify(element.content)
  if not element.classes:includes('unnumbered') and title ~= 'Acknowledgements' and title ~= 'References' then
    sections[element.level] = sections[element.level] + 1
    for level = element.level + 1, #sections do
      sections[level] = 0
    end
    local number = table.concat(sections, '.', 1, element.level)
    element.content:insert(1, pandoc.Space())
    element.content:insert(1, pandoc.Str(number .. '.'))
    style = 'Heading ' .. element.level
  end
  return pandoc.Div({pandoc.Para(element.content)}, {['custom-style'] = style})
end

---@type integer
local figures = 0
---@type integer
local tables = 0

---@param caption pandoc.Caption
---@param label string
---@return pandoc.Div
local function labelled_caption(caption, label)
  local inlines = pandoc.utils.blocks_to_inlines(caption.long)
  inlines:insert(1, pandoc.Space())
  inlines:insert(1, pandoc.Strong({pandoc.Str(label)}))
  return pandoc.Div({pandoc.Para(inlines)}, {['custom-style'] = 'CaptionLong'})
end

---@param element pandoc.Figure
---@return pandoc.Blocks
function Figure(element)
  figures = figures + 1
  local blocks = element.content
  blocks:insert(labelled_caption(element.caption, 'Figure ' .. figures .. '.'))
  return blocks
end

---@param element pandoc.Table
---@return pandoc.Blocks
function Table(element)
  tables = tables + 1
  local caption = labelled_caption(element.caption, 'Table ' .. tables .. '.')
  element.caption = pandoc.Caption()
  return {caption, element}
end

---@param element pandoc.Image
---@return pandoc.Image
function Image(element)
  element.attributes.width = '12.4cm'
  element.attributes.height = nil
  return element
end

-- Unknown LaTeX must not silently disappear from a submission export.
---@param element pandoc.RawBlock|pandoc.RawInline
local function reject_raw(element)
  error('Unconverted ' .. element.format .. ': ' .. element.text)
end
RawBlock = reject_raw
RawInline = reject_raw
