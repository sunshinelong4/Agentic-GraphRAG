import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { motion } from 'motion/react';
import { Send, Sparkles, CheckCircle2, ShieldCheck, Quote, X, FileText, Zap, Eye, BookOpen, Tag, Layers, GitBranch, Info, Network, ArrowRight, Maximize2, FileSearch } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { mockDocuments } from '../../data/mockDocuments';
import { apiService, ApiError, QAResponse, QASource } from '../../services/api';

// 实体类型对应的颜色
const ENTITY_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  '人名': { bg: 'bg-blue-100', text: 'text-blue-800', border: 'border-blue-300' },
  '地名': { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-300' },
  '组织': { bg: 'bg-purple-100', text: 'text-purple-800', border: 'border-purple-300' },
  '时间': { bg: 'bg-orange-100', text: 'text-orange-800', border: 'border-orange-300' },
  '金额': { bg: 'bg-yellow-100', text: 'text-yellow-800', border: 'border-yellow-300' },
  '药品': { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-300' },
  '疾病': { bg: 'bg-pink-100', text: 'text-pink-800', border: 'border-pink-300' },
  '症状': { bg: 'bg-indigo-100', text: 'text-indigo-800', border: 'border-indigo-300' },
  '检查项目': { bg: 'bg-cyan-100', text: 'text-cyan-800', border: 'border-cyan-300' },
  '手术': { bg: 'bg-rose-100', text: 'text-rose-800', border: 'border-rose-300' },
  'default': { bg: 'bg-gray-100', text: 'text-gray-800', border: 'border-gray-300' },
};

// 获取实体颜色
const getEntityColor = (entityType: string) => {
  return ENTITY_COLORS[entityType] || ENTITY_COLORS['default'];
};

interface QuestionAnswerProps {
  onPrevious: () => void;
  selectedDocument: string | null;
}

interface Citation {
  text: string;
  location: string;
  page?: number;
  paragraph?: number;
  confidence: number;
  chunk_type?: string;
  extraction_class?: string;
  char_interval?: {
    start_pos: number;
    end_pos: number;
  };
  attributes?: Record<string, any>;
}

interface MatchedEntity {
  text: string;
  entity_type: string;
  confidence: number;
  start_pos: number;
  end_pos: number;
}

// 结构化答案格式
interface StructuredAnswer {
  summary: string;
  items: Array<{
    title: string;
    highlights: string[];
    detail: string;
  }>;
  conclusion?: string;
}

interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  structuredContent?: StructuredAnswer;  // 结构化内容
  citations?: Citation[];
  confidence?: number;
  timestamp: Date;
  entities?: MatchedEntity[];
}

// 节点颜色配置
const NODE_COLORS = [
  { fill: '#3b82f6', stroke: '#1e40af', bg: 'rgba(59, 130, 246, 0.15)' },  // 蓝色
  { fill: '#8b5cf6', stroke: '#6b21a8', bg: 'rgba(139, 92, 246, 0.15)' },  // 紫色
  { fill: '#10b981', stroke: '#047857', bg: 'rgba(16, 185, 129, 0.15)' },  // 绿色
  { fill: '#f59e0b', stroke: '#b45309', bg: 'rgba(245, 158, 11, 0.15)' },  // 橙色
  { fill: '#ef4444', stroke: '#b91c1c', bg: 'rgba(239, 68, 68, 0.15)' },   // 红色
  { fill: '#06b6d4', stroke: '#0e7490', bg: 'rgba(6, 182, 212, 0.15)' },   // 青色
];

// 知识图谱可视化组件 - 支持链式关系展示
const KnowledgeGraphVisualization: React.FC<{
  attributes: Record<string, any>;
  extractionClass?: string;
  contentText?: string;
}> = ({ attributes, extractionClass, contentText }) => {
  // 构建图节点的多种策略
  let nodes: { label: string; type: string }[] = [];

  // 策略1: 解析 mechanism_group 字段，格式如: "中药-通路-HSC"
  const mechanismGroup = attributes['mechanism_group'] || '';
  if (mechanismGroup) {
    const parts = mechanismGroup.split('-').filter((n: string) => n.trim());
    nodes = parts.map((p: string, i: number) => ({ label: p, type: `节点${i + 1}` }));
  }

  // 策略2: 从主体/关系/对象属性构建
  if (nodes.length === 0) {
    const subject1 = attributes['主体1'] || attributes['subject1'] || attributes['实体1'] || '';
    const relation = attributes['关系'] || attributes['relation'] || '';
    const subject2 = attributes['主体2'] || attributes['subject2'] || attributes['实体2'] || attributes['对象'] || '';

    if (subject1) nodes.push({ label: subject1, type: '主体' });
    if (relation) nodes.push({ label: relation, type: '关系' });
    if (subject2) nodes.push({ label: subject2, type: '对象' });
  }

  // 策略3: 将所有属性展示为中心放射图 - 以 extraction_class 为中心
  if (nodes.length < 2 && Object.keys(attributes).length > 0) {
    // 中心节点
    const centerLabel = extractionClass || contentText?.substring(0, 15) || '实体';
    nodes.push({ label: centerLabel, type: '核心' });

    // 从 attributes 中提取属性节点（排除一些内部字段）
    const excludeKeys = ['char_interval', 'start_pos', 'end_pos', 'confidence', 'score'];
    Object.entries(attributes).forEach(([key, value]) => {
      if (!excludeKeys.includes(key) && value && typeof value === 'string' && value.length < 30) {
        nodes.push({ label: `${key}: ${value}`, type: '属性' });
      } else if (!excludeKeys.includes(key) && value && typeof value === 'string') {
        nodes.push({ label: `${key}: ${value.substring(0, 20)}...`, type: '属性' });
      }
    });
  }

  // 策略4: 最后使用 extractionClass 和 contentText
  if (nodes.length < 2 && extractionClass && contentText) {
    nodes = [
      { label: extractionClass, type: '类型' },
      { label: contentText.length > 20 ? contentText.substring(0, 20) + '...' : contentText, type: '内容' }
    ];
  }

  // 如果节点太少，不渲染
  if (nodes.length < 2) return null;

  // 判断图形类型：链式还是星形
  const isChainGraph = mechanismGroup || nodes.every(n => n.type !== '属性');
  const isStarGraph = !isChainGraph && nodes.some(n => n.type === '核心');

  // 计算布局
  const svgWidth = 380;
  const svgHeight = isStarGraph ? 200 : (nodes.length > 4 ? 180 : 130);
  const nodeWidth = 80;
  const nodeHeight = 34;
  const horizontalGap = 25;

  // 链式图布局：水平排列，可能换行
  const maxNodesPerRow = 4;
  const rows = Math.ceil(nodes.length / maxNodesPerRow);
  const nodesPerRow = Math.ceil(nodes.length / rows);

  const getChainNodePosition = (index: number) => {
    const row = Math.floor(index / nodesPerRow);
    const col = index % nodesPerRow;
    const nodesInThisRow = Math.min(nodesPerRow, nodes.length - row * nodesPerRow);
    const rowWidth = nodesInThisRow * nodeWidth + (nodesInThisRow - 1) * horizontalGap;
    const startX = (svgWidth - rowWidth) / 2;

    return {
      x: startX + col * (nodeWidth + horizontalGap) + nodeWidth / 2,
      y: 40 + row * 60,
    };
  };

  // 星形图布局：中心节点 + 周围属性节点
  const getStarNodePosition = (index: number) => {
    const centerX = svgWidth / 2;
    const centerY = svgHeight / 2;

    if (index === 0) {
      // 中心节点
      return { x: centerX, y: centerY };
    }

    // 周围节点，环形分布
    const totalSatellites = nodes.length - 1;
    const angleStep = (2 * Math.PI) / totalSatellites;
    const angle = -Math.PI / 2 + (index - 1) * angleStep; // 从顶部开始
    const radius = Math.min(svgWidth, svgHeight) / 2 - 50;

    return {
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    };
  };

  const getNodePosition = isStarGraph ? getStarNodePosition : getChainNodePosition;

  return (
    <div className="relative w-full bg-gradient-to-br from-slate-50 via-purple-50 to-blue-50 rounded-xl border border-purple-200 overflow-hidden">
      {/* 背景网格 */}
      <div className="absolute inset-0 opacity-20">
        <svg width="100%" height="100%">
          <defs>
            <pattern id="kg-grid" width="15" height="15" patternUnits="userSpaceOnUse">
              <circle cx="1" cy="1" r="0.5" fill="#8b5cf6" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#kg-grid)" />
        </svg>
      </div>

      {/* 知识图谱 SVG */}
      <svg width="100%" height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="relative z-10">
        <defs>
          {/* 渐变定义 */}
          <linearGradient id="kg-line-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#8b5cf6" />
            <stop offset="100%" stopColor="#3b82f6" />
          </linearGradient>
          <linearGradient id="kg-radial-gradient" x1="50%" y1="50%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#a855f7" />
            <stop offset="100%" stopColor="#6366f1" />
          </linearGradient>
          {/* 箭头标记 */}
          <marker id="kg-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6366f1" />
          </marker>
          {/* 发光效果 */}
          <filter id="kg-glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
          <filter id="kg-center-glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* 绘制连接线 */}
        {isStarGraph ? (
          // 星形图连线：从中心向外辐射
          nodes.slice(1).map((_, index) => {
            const from = getNodePosition(0); // 中心
            const to = getNodePosition(index + 1); // 外围节点
            const dx = to.x - from.x;
            const dy = to.y - from.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const offsetRatio = 45 / dist;

            return (
              <motion.line
                key={`line-${index}`}
                x1={from.x + dx * offsetRatio}
                y1={from.y + dy * offsetRatio}
                x2={to.x - dx * offsetRatio * 0.8}
                y2={to.y - dy * offsetRatio * 0.8}
                stroke="url(#kg-radial-gradient)"
                strokeWidth="2"
                strokeDasharray="4,2"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 0.7 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              />
            );
          })
        ) : (
          // 链式图连线
          nodes.map((_, index) => {
            if (index === nodes.length - 1) return null;
            const from = getNodePosition(index);
            const to = getNodePosition(index + 1);

            // 判断是否换行
            const fromRow = Math.floor(index / nodesPerRow);
            const toRow = Math.floor((index + 1) / nodesPerRow);

            if (fromRow === toRow) {
              // 同一行，直线连接
              return (
                <motion.line
                  key={`line-${index}`}
                  x1={from.x + nodeWidth / 2 - 5}
                  y1={from.y}
                  x2={to.x - nodeWidth / 2 + 5}
                  y2={to.y}
                  stroke="url(#kg-line-gradient)"
                  strokeWidth="2.5"
                  markerEnd="url(#kg-arrow)"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.4, delay: index * 0.15 }}
                />
              );
            } else {
              // 换行，用曲线连接
              const midY = (from.y + to.y) / 2;
              return (
                <motion.path
                  key={`line-${index}`}
                  d={`M ${from.x + nodeWidth / 2 - 5} ${from.y}
                      Q ${from.x + nodeWidth / 2 + 20} ${midY} ${to.x - nodeWidth / 2 + 5} ${to.y}`}
                  stroke="url(#kg-line-gradient)"
                  strokeWidth="2.5"
                  fill="none"
                  markerEnd="url(#kg-arrow)"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.4, delay: index * 0.15 }}
                />
              );
            }
          })
        )}

        {/* 绘制节点 */}
        {nodes.map((node, index) => {
          const pos = getNodePosition(index);
          const color = NODE_COLORS[index % NODE_COLORS.length];
          const displayText = node.label.length > 8 ? node.label.substring(0, 7) + '..' : node.label;
          const isCenterNode = isStarGraph && index === 0;
          const currentNodeWidth = isCenterNode ? nodeWidth + 10 : nodeWidth;
          const currentNodeHeight = isCenterNode ? nodeHeight + 6 : nodeHeight;

          return (
            <motion.g
              key={`node-${index}`}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.3, delay: index * 0.1, type: "spring", stiffness: 300 }}
            >
              {/* 节点背景发光 */}
              <ellipse
                cx={pos.x}
                cy={pos.y}
                rx={currentNodeWidth / 2 + 4}
                ry={currentNodeHeight / 2 + 4}
                fill={color.fill}
                opacity={isCenterNode ? 0.25 : 0.15}
                filter={isCenterNode ? "url(#kg-center-glow)" : "url(#kg-glow)"}
              />
              {/* 节点主体 */}
              <rect
                x={pos.x - currentNodeWidth / 2}
                y={pos.y - currentNodeHeight / 2}
                width={currentNodeWidth}
                height={currentNodeHeight}
                rx="8"
                fill={isCenterNode ? color.fill : "white"}
                stroke={color.fill}
                strokeWidth={isCenterNode ? 3 : 2}
              />
              {/* 节点文字 */}
              <text
                x={pos.x}
                y={pos.y + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={isCenterNode ? 11 : 10}
                fontWeight="600"
                fill={isCenterNode ? "white" : color.stroke}
              >
                {displayText}
              </text>
              {/* 节点类型标记 */}
              {node.type && !isCenterNode && (
                <>
                  <rect
                    x={pos.x - 12}
                    y={pos.y - currentNodeHeight / 2 - 10}
                    width={24}
                    height={12}
                    rx="6"
                    fill={color.fill}
                  />
                  <text
                    x={pos.x}
                    y={pos.y - currentNodeHeight / 2 - 4}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize="7"
                    fontWeight="bold"
                    fill="white"
                  >
                    {node.type.length > 4 ? node.type.substring(0, 3) : node.type}
                  </text>
                </>
              )}
              {/* 中心节点的特殊标记 */}
              {isCenterNode && (
                <circle
                  cx={pos.x + currentNodeWidth / 2 - 5}
                  cy={pos.y - currentNodeHeight / 2 + 5}
                  r="6"
                  fill="#fbbf24"
                  stroke="white"
                  strokeWidth="2"
                />
              )}
            </motion.g>
          );
        })}
      </svg>

      {/* 图例和说明 */}
      <div className="px-3 py-2 border-t border-purple-100 bg-white/50">
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-purple-600 font-medium">
            {isStarGraph ? '🌟 属性网络' : '🔗 关系链路'}: {nodes.length} 个节点
          </span>
          <span className="text-gray-500">
            {extractionClass && `类型: ${extractionClass}`}
          </span>
        </div>
        {/* 节点图例 */}
        <div className="flex flex-wrap gap-1.5 mt-1.5">
          {nodes.slice(0, 5).map((node, index) => {
            const color = NODE_COLORS[index % NODE_COLORS.length];
            return (
              <span
                key={index}
                className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px]"
                style={{ backgroundColor: color.bg, color: color.stroke }}
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color.fill }}></span>
                {node.label.length > 10 ? node.label.substring(0, 10) + '..' : node.label}
              </span>
            );
          })}
          {nodes.length > 5 && (
            <span className="text-[9px] text-gray-400">+{nodes.length - 5} 更多</span>
          )}
        </div>
      </div>
    </div>
  );
};

// 高亮颜色配置
const HIGHLIGHT_COLORS = [
  { bg: 'bg-amber-200', text: 'text-amber-900', name: '黄色' },
  { bg: 'bg-blue-200', text: 'text-blue-900', name: '蓝色' },
  { bg: 'bg-green-200', text: 'text-green-900', name: '绿色' },
  { bg: 'bg-purple-200', text: 'text-purple-900', name: '紫色' },
  { bg: 'bg-pink-200', text: 'text-pink-900', name: '粉色' },
  { bg: 'bg-cyan-200', text: 'text-cyan-900', name: '青色' },
  { bg: 'bg-orange-200', text: 'text-orange-900', name: '橙色' },
  { bg: 'bg-red-200', text: 'text-red-900', name: '红色' },
];

// 多关键词高亮组件 - 用于源文档显示
const MultiKeywordHighlight: React.FC<{
  text: string;
  keywords: Array<{ text: string; label?: string }>;
}> = ({ text, keywords }) => {
  if (!text) return null;
  if (!keywords || keywords.length === 0) {
    return <span>{text}</span>;
  }

  // 找到所有关键词的位置
  const matches: Array<{ start: number; end: number; keyword: string; colorIdx: number }> = [];

  keywords.forEach((kw, kwIdx) => {
    if (!kw.text || kw.text.length < 2) return; // 忽略太短的关键词
    let searchStart = 0;
    const kwText = kw.text;
    while (searchStart < text.length) {
      const idx = text.indexOf(kwText, searchStart);
      if (idx === -1) break;
      matches.push({
        start: idx,
        end: idx + kwText.length,
        keyword: kwText,
        colorIdx: kwIdx % HIGHLIGHT_COLORS.length
      });
      searchStart = idx + 1;
    }
  });

  // 按位置排序
  matches.sort((a, b) => a.start - b.start);

  // 合并重叠区间
  const mergedMatches: typeof matches = [];
  for (const m of matches) {
    if (mergedMatches.length === 0) {
      mergedMatches.push(m);
    } else {
      const last = mergedMatches[mergedMatches.length - 1];
      if (m.start <= last.end) {
        // 重叠，扩展
        last.end = Math.max(last.end, m.end);
      } else {
        mergedMatches.push(m);
      }
    }
  }

  // 构建结果
  const result: React.ReactNode[] = [];
  let lastEnd = 0;

  mergedMatches.forEach((m, idx) => {
    // 添加前面的普通文本
    if (m.start > lastEnd) {
      result.push(<span key={`text-${idx}`}>{text.substring(lastEnd, m.start)}</span>);
    }
    // 添加高亮文本
    const color = HIGHLIGHT_COLORS[m.colorIdx];
    result.push(
      <mark
        key={`hl-${idx}`}
        className={`${color.bg} ${color.text} px-0.5 rounded font-medium`}
        title={m.keyword}
      >
        {text.substring(m.start, m.end)}
      </mark>
    );
    lastEnd = m.end;
  });

  // 添加剩余文本
  if (lastEnd < text.length) {
    result.push(<span key="text-end">{text.substring(lastEnd)}</span>);
  }

  return <>{result}</>;
};

// 原文高亮显示组件
const SourceTextHighlight: React.FC<{
  fullText: string;
  charInterval?: { start_pos: number; end_pos: number };
  highlightText?: string;
}> = ({ fullText, charInterval, highlightText }) => {
  if (!fullText) return null;

  // 计算高亮范围（显示前后各100个字符的上下文）
  const contextSize = 100;
  let startPos = 0;
  let endPos = fullText.length;
  let highlightStart = 0;
  let highlightEnd = 0;

  if (charInterval) {
    highlightStart = charInterval.start_pos;
    highlightEnd = charInterval.end_pos;
    startPos = Math.max(0, highlightStart - contextSize);
    endPos = Math.min(fullText.length, highlightEnd + contextSize);
  } else if (highlightText) {
    const idx = fullText.indexOf(highlightText);
    if (idx >= 0) {
      highlightStart = idx;
      highlightEnd = idx + highlightText.length;
      startPos = Math.max(0, highlightStart - contextSize);
      endPos = Math.min(fullText.length, highlightEnd + contextSize);
    }
  }

  const beforeText = fullText.substring(startPos, highlightStart);
  const matchText = fullText.substring(highlightStart, highlightEnd);
  const afterText = fullText.substring(highlightEnd, endPos);

  return (
    <div className="relative">
      {/* 原文显示区域 */}
      <div className="text-sm leading-relaxed text-gray-700 font-mono bg-gray-50 p-3 rounded-lg border border-gray-200 max-h-48 overflow-y-auto">
        {startPos > 0 && <span className="text-gray-400">...</span>}
        <span>{beforeText}</span>
        <mark className="bg-amber-300 text-amber-900 px-0.5 rounded font-semibold">
          {matchText}
        </mark>
        <span>{afterText}</span>
        {endPos < fullText.length && <span className="text-gray-400">...</span>}
      </div>

      {/* 位置指示器 */}
      {charInterval && (
        <div className="absolute -top-2 -right-2 px-2 py-0.5 bg-amber-500 text-white text-[10px] font-medium rounded-full shadow">
          字符 {charInterval.start_pos}-{charInterval.end_pos}
        </div>
      )}
    </div>
  );
};

export function QuestionAnswer({ selectedDocument }: QuestionAnswerProps) {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null); // 当前选中引用所属的消息ID
  const [viewMode, setViewMode] = useState<'traditional' | 'structured'>('structured');
  const [extractedEntities, setExtractedEntities] = useState<Array<{text: string; entity_type: string}>>([]);
  const [selectedEntity, setSelectedEntity] = useState<MatchedEntity | null>(null);
  const [useKnowledgeBase, setUseKnowledgeBase] = useState(false);
  const [originalText, setOriginalText] = useState<string>(''); // 原文内容
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const citationRefs = useRef<Map<string, HTMLButtonElement>>(new Map());

  // 获取当前选中消息的实体（用于证据面板显示）
  const currentMessageEntities = useMemo(() => {
    if (!selectedMessageId) return [];
    const msg = messages.find(m => m.id === selectedMessageId);

    // 优先使用消息中已匹配的实体
    if (msg?.entities && msg.entities.length > 0) {
      // 去重：按 text 去重
      const uniqueEntities = new Map<string, {text: string; entity_type: string}>();
      msg.entities.forEach(e => {
        if (!uniqueEntities.has(e.text)) {
          uniqueEntities.set(e.text, { text: e.text, entity_type: e.entity_type });
        }
      });
      return Array.from(uniqueEntities.values());
    }

    // 如果消息没有预匹配的实体，尝试从答案文本中动态匹配 extractedEntities
    if (msg?.content && extractedEntities.length > 0) {
      const matchedInContent: Array<{text: string; entity_type: string}> = [];
      const seenTexts = new Set<string>();

      for (const entity of extractedEntities) {
        if (entity.text && msg.content.includes(entity.text) && !seenTexts.has(entity.text)) {
          seenTexts.add(entity.text);
          matchedInContent.push({
            text: entity.text,
            entity_type: entity.entity_type
          });
        }
      }

      if (matchedInContent.length > 0) {
        return matchedInContent;
      }
    }

    // 最后回退到所有已提取的实体
    return extractedEntities;
  }, [selectedMessageId, messages, extractedEntities]);

  // 加载已提取的实体信息和原文内容
  useEffect(() => {
    const loadExtractedEntities = () => {
      try {
        // 从 sessionStorage 读取提取结果（与 InformationExtraction 组件保存的键名一致）
        const extractionsStr = sessionStorage.getItem('extractedItems');
        console.log('[QuestionAnswer] ========== 加载实体信息 ==========');
        console.log('[QuestionAnswer] sessionStorage extractedItems 原始数据:', extractionsStr?.substring(0, 500));
        if (extractionsStr) {
          const extractions = JSON.parse(extractionsStr);
          console.log('[QuestionAnswer] 解析后的提取项数量:', extractions.length);
          console.log('[QuestionAnswer] 前 5 个提取项:', extractions.slice(0, 5));
          // 提取实体列表 - 字段名来自 InformationExtraction 组件:
          // { type: extraction_class, value: extraction_text, ... }
          const entities = extractions.map((ext: any) => ({
            text: ext.value || ext.text || ext.extraction_text,
            entity_type: ext.type || ext.extraction_class,
          }));
          console.log('[QuestionAnswer] 转换后的实体列表:', entities);
          console.log('[QuestionAnswer] 实体示例:', entities.slice(0, 5).map((e: any) => `${e.text}(${e.entity_type})`).join(', '));
          console.log('[QuestionAnswer] =====================================');
          setExtractedEntities(entities);
        } else {
          console.log('[QuestionAnswer] 警告: sessionStorage 中没有 extractedItems 数据');
          console.log('[QuestionAnswer] 请先在"信息提取"步骤中进行实体提取');
        }

        // 加载原文内容（用于溯源高亮）
        const pastedText = sessionStorage.getItem('pastedText');
        if (pastedText) {
          setOriginalText(pastedText);
          console.log('[QuestionAnswer] 已加载原文内容，长度:', pastedText.length);
        }
      } catch (e) {
        console.error('[QuestionAnswer] 加载实体失败:', e);
      }
    };
    loadExtractedEntities();
  }, []);

  // 默认直接使用知识库模式（不显示欢迎页面）
  useEffect(() => {
    if (!selectedDocument && !useKnowledgeBase) {
      setUseKnowledgeBase(true);
    }
  }, [selectedDocument, useKnowledgeBase]);

  // 判断是否是用户上传的文档或使用知识库模式
  const isPastedText = selectedDocument === 'pasted-text' || selectedDocument === 'pdf-parsed';
  const isKnowledgeBaseMode = useKnowledgeBase && !selectedDocument;
  const doc = !isPastedText && selectedDocument ? mockDocuments[selectedDocument as keyof typeof mockDocuments] : null;

  // Auto-scroll to bottom - 只在新消息添加时滚动，不在流式更新时滚动
  const prevMessageCountRef = useRef(0);
  useEffect(() => {
    // 只有当消息数量增加时才滚动（新消息添加），而不是内容更新时
    if (messages.length > prevMessageCountRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
    prevMessageCountRef.current = messages.length;
  }, [messages.length]);

  // Example questions
  const exampleQuestions = {
    contract: [
      '合同总金额是多少？',
      '谁是甲方和乙方？',
      '支付方式是什么？',
    ],
    research: [
      '海洋污染的主要原因是什么？',
      '微塑料浓度是多少？',
      '谁进行了这项研究？',
    ],
    medical: [
      '患者的主要诊断是什么？',
      '推荐使用哪些药物？',
      '手术风险有哪些？',
    ],
    'pasted-text': [
      '请总结这段文本的主要内容',
      '文本中提到了哪些关键信息？',
      '这段文本的结论是什么？',
    ],
    'pdf-parsed': [
      '请总结这个文档的主要内容',
      '文档中提到了哪些关键信息？',
      '这个文档的重点是什么？',
    ],
    'knowledge-base': [
      '知识库中有哪些文档？',
      '请介绍一下主要的法律条款',
      '文档中涉及哪些重要概念？',
    ],
  };

  const questions = isKnowledgeBaseMode
    ? exampleQuestions['knowledge-base']
    : (exampleQuestions[selectedDocument as keyof typeof exampleQuestions] || []);

  // Mock citations database
  const getCitationsForQuestion = (questionText: string): Citation[] => {
    const citationMap: Record<string, Citation[]> = {
      '合同总金额是多少？': [
        {
          text: '合同总金额为人民币500万元整',
          location: '第五条 合同金额与支付方式',
          page: 3,
          paragraph: 2,
          confidence: 0.98,
        },
        {
          text: '甲乙双方经友好协商，确定本合同项目总价款为人民币伍佰万元整（¥5,000,000.00）',
          location: '第五条 第1款',
          page: 3,
          paragraph: 3,
          confidence: 0.96,
        },
      ],
      '谁是甲方和乙方？': [
        {
          text: '甲方：科技创新发展有限公司',
          location: '合同首部',
          page: 1,
          paragraph: 1,
          confidence: 0.99,
        },
        {
          text: '乙方：智能解决方案股份有限公司',
          location: '合同首部',
          page: 1,
          paragraph: 2,
          confidence: 0.99,
        },
      ],
      '海洋污染的主要原因是什么？': [
        {
          text: '研究表明，海洋污染的主要来源为塑料废弃物和工业排放',
          location: '第二章 研究背景',
          page: 2,
          paragraph: 1,
          confidence: 0.94,
        },
        {
          text: '每年约有800万吨塑料进入海洋环境',
          location: '第三章 数据分析',
          page: 4,
          paragraph: 3,
          confidence: 0.91,
        },
      ],
      '患者的主要诊断是什么？': [
        {
          text: '初步诊断：冠状动脉粥样硬化性心脏病',
          location: '诊断结论',
          page: 2,
          paragraph: 1,
          confidence: 0.97,
        },
        {
          text: '合并症：高血压病（2级，高危）',
          location: '诊断结论',
          page: 2,
          paragraph: 2,
          confidence: 0.95,
        },
      ],
    };

    return citationMap[questionText] || [];
  };

  // Convert QASource to Citation format
  const sourceToCitation = (source: QASource, index: number): Citation => ({
    text: source.content_preview,
    location: source.doc_title,
    page: (source.chunk_index ?? index) + 1,
    paragraph: 1,
    confidence: source.score,
    chunk_type: source.chunk_type,
    extraction_class: source.extraction_class,
    char_interval: source.char_interval,
    attributes: source.attributes,
  });

  // 智能解析文本为结构化格式
  const parseTextToStructured = (text: string): StructuredAnswer | null => {
    try {
      // 1. 首先尝试解析 JSON（如果LLM输出了JSON）
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        try {
          const parsed = JSON.parse(jsonMatch[0]);
          if (parsed.summary && parsed.items) {
            return parsed as StructuredAnswer;
          }
        } catch {}
      }

      // 2. 清理文本：移除"来源："相关内容
      let cleanText = text
        .replace(/来源[：:][^\n]*/g, '')
        .replace(/\*来源[：:][^\n]*/g, '')
        .replace(/参考[：:][^\n]*/g, '')
        .replace(/\n{3,}/g, '\n\n')
        .trim();

      // 3. 分割成行
      const lines = cleanText.split('\n').filter(line => line.trim());
      if (lines.length === 0) return null;

      // 4. 收集摘要内容（开头的非列表段落）
      const summaryParts: string[] = [];
      let startIdx = 0;

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        // 跳过标题行
        if (line.startsWith('#')) {
          continue;
        }

        // 检测是否是列表项开头
        if (line.match(/^(?:[\d]+[\.、\)]|[•\-\*])\s+/)) {
          startIdx = i;
          break;
        }

        // 非列表行，作为摘要的一部分
        const cleanLine = line.replace(/^\*+\s*/, '').replace(/\*+$/, '').trim();
        if (cleanLine.length > 5 && !cleanLine.includes('来源') && !cleanLine.includes('参考')) {
          summaryParts.push(cleanLine);
        }

        startIdx = i + 1;
      }

      // 合并摘要
      let summary = summaryParts.join(' ').replace(/\*+/g, '').trim();
      if (summary.length > 300) {
        summary = summary.substring(0, 300) + '...';
      }

      // 5. 解析列表项
      const items: Array<{title: string; highlights: string[]; detail: string}> = [];
      let currentItem: {title: string; highlights: string[]; detail: string} | null = null;

      for (let i = startIdx; i < lines.length; i++) {
        const line = lines[i].trim();

        // 跳过空行和来源行
        if (!line || line.includes('来源') || line.includes('参考')) continue;

        // 检测列表项开头: 1. 2. • - * 等
        const listMatch = line.match(/^(?:[\d]+[\.、\)]|[•\-\*])\s*(.+)/);
        if (listMatch) {
          // 保存之前的项
          if (currentItem && currentItem.title) {
            items.push(currentItem);
          }

          // 解析新项
          let content = listMatch[1];

          // 提取【】中的关键词
          const highlights: string[] = [];
          const highlightMatches = content.match(/【([^】]+)】/g);
          if (highlightMatches) {
            highlightMatches.forEach(m => {
              highlights.push(m.replace(/[【】]/g, ''));
            });
          }

          // 尝试分离标题和详情（通过：或:）
          const colonIdx = content.search(/[：:]/);
          let title = content;
          let detail = '';

          if (colonIdx > 0 && colonIdx < 30) {
            title = content.substring(0, colonIdx).replace(/\*+/g, '').trim();
            detail = content.substring(colonIdx + 1).replace(/\*+/g, '').trim();
          } else {
            // 移除markdown格式
            title = content.replace(/\*+([^*]+)\*+/g, '$1').trim();
            if (title.length > 50) {
              detail = title;
              title = title.substring(0, 30) + '...';
            }
          }

          // 清理标题中的【】
          title = title.replace(/【[^】]+】/g, '').trim();

          currentItem = {
            title: title || '要点',
            highlights: highlights.length > 0 ? highlights : extractKeywords(content),
            detail: detail || ''
          };
        } else if (currentItem) {
          // 非列表行，追加到当前项的detail
          currentItem.detail += (currentItem.detail ? ' ' : '') + line.replace(/\*+/g, '');
        }
      }

      // 添加最后一项
      if (currentItem && currentItem.title) {
        items.push(currentItem);
      }

      // 6. 如果没有解析出items，尝试按段落分割
      if (items.length === 0) {
        const paragraphs = cleanText.split(/\n\n+/).filter(p => p.trim() && !p.includes('来源'));
        paragraphs.forEach((p, idx) => {
          if (idx === 0 && !summary) {
            summary = p.substring(0, 200);
            return;
          }
          const cleanP = p.replace(/\*+/g, '').trim();
          if (cleanP.length > 10) {
            items.push({
              title: `要点 ${items.length + 1}`,
              highlights: extractKeywords(cleanP),
              detail: cleanP
            });
          }
        });
      }

      // 7. 如果还是没有items，把整个文本作为summary
      if (items.length === 0 && cleanText.length > 20) {
        if (!summary) {
          summary = cleanText.substring(0, 500);
        }
        // 不创建额外的item，直接返回只有summary的结果
      }

      // 8. 如果没有summary但有items，用第一个item生成summary
      if (!summary && items.length > 0) {
        summary = items[0].title + (items[0].detail ? '：' + items[0].detail.substring(0, 100) : '');
      }

      return {
        summary: summary || '以下是相关信息',
        items,
        conclusion: null
      };
    } catch (e) {
      console.error('解析结构化内容失败:', e);
      return null;
    }
  };

  // 从文本中提取关键词
  const extractKeywords = (text: string): string[] => {
    // 提取中文专有名词（2-6个字的词组）
    const keywords: string[] = [];

    // 匹配引号、书名号中的内容
    const quotedMatches = text.match(/[""「」『』《》]([^""「」『』《》]+)[""「」『』《》]/g);
    if (quotedMatches) {
      quotedMatches.slice(0, 2).forEach(m => {
        keywords.push(m.replace(/[""「」『』《》]/g, ''));
      });
    }

    // 匹配特定模式（药物名、疾病名等）
    const termMatches = text.match(/[\u4e00-\u9fa5]{2,8}(?:素|酶|剂|药|方|汤|丸|散|胶囊|片|化|症|病|炎|癌)/g);
    if (termMatches) {
      termMatches.slice(0, 3 - keywords.length).forEach(m => {
        if (!keywords.includes(m)) keywords.push(m);
      });
    }

    // 如果关键词不够，提取一些常见的中文词汇
    if (keywords.length < 2) {
      const commonMatches = text.match(/[\u4e00-\u9fa5]{3,6}/g);
      if (commonMatches) {
        const unique = [...new Set(commonMatches)];
        unique.slice(0, 3 - keywords.length).forEach(m => {
          if (!keywords.includes(m) && m.length >= 3) keywords.push(m);
        });
      }
    }

    return keywords.slice(0, 4);
  };

  // Generate AI answer - uses streaming API
  const generateAnswer = async (questionText: string) => {
    // Add user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: questionText,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    setQuestion('');
    setIsGenerating(true);
    const aiMessageId = `ai-${Date.now()}`;

    try {
      // Use streaming API
      let answerContent = '';
      let citations: Citation[] = [];
      let sources: QASource[] = [];
      let matchedEntities: MatchedEntity[] = [];

      // Add empty AI message that will be updated
      const aiMessage: Message = {
        id: aiMessageId,
        type: 'ai',
        content: '',
        citations: [],
        confidence: 0,
        timestamp: new Date(),
        entities: [],
      };
      setMessages(prev => [...prev, aiMessage]);

      // Stream the response with entities
      for await (const event of apiService.askQuestionStream({
        question: questionText,
        top_k: 5,
        entities: extractedEntities,
      })) {
        if (event.type === 'sources') {
          // Received sources
          sources = event.data;
          citations = sources.map((s, i) => sourceToCitation(s, i));
          const avgConfidence = citations.length > 0
            ? citations.reduce((sum, c) => sum + c.confidence, 0) / citations.length
            : 0;

          setMessages(prev => prev.map(msg =>
            msg.id === aiMessageId
              ? { ...msg, citations, confidence: avgConfidence }
              : msg
          ));
        } else if (event.type === 'chunk') {
          // Received content chunk
          answerContent += event.data.content;
          setMessages(prev => prev.map(msg =>
            msg.id === aiMessageId
              ? { ...msg, content: answerContent }
              : msg
          ));
        } else if (event.type === 'matched_entities') {
          // Received matched entities with positions
          matchedEntities = event.data;
          setMessages(prev => prev.map(msg =>
            msg.id === aiMessageId
              ? { ...msg, entities: matchedEntities }
              : msg
          ));
        } else if (event.type === 'done') {
          // Stream completed - 智能解析文本为结构化格式
          const structured = parseTextToStructured(answerContent);
          if (structured) {
            setMessages(prev => prev.map(msg =>
              msg.id === aiMessageId
                ? { ...msg, structuredContent: structured }
                : msg
            ));
          }
          setIsGenerating(false);
          return;
        } else if (event.type === 'error') {
          // Error occurred
          console.error('Stream error:', event.data.error);
          setMessages(prev => prev.map(msg =>
            msg.id === aiMessageId
              ? { ...msg, content: `错误: ${event.data.error}` }
              : msg
          ));
          setIsGenerating(false);
          return;
        }
      }

      setIsGenerating(false);
      return;
    } catch (error) {
      if (error instanceof ApiError) {
        setMessages(prev => prev.map(msg => msg.id === aiMessageId ? { ...msg, content: error.message } : msg));
        setIsGenerating(false);
        return;
      }
      console.log('Streaming API failed, falling back to regular API:', error);
      // Fall back to regular API
      try {
        const response: QAResponse = await apiService.askQuestion({
          question: questionText,
          top_k: 5,
        });

        if (response.success && response.answer) {
          const citations: Citation[] = response.sources.map((s, i) => sourceToCitation(s, i));
          const avgConfidence = citations.length > 0
            ? citations.reduce((sum, c) => sum + c.confidence, 0) / citations.length
            : 0;

          const aiMessage: Message = {
            id: `ai-${Date.now()}`,
            type: 'ai',
            content: response.answer,
            citations,
            confidence: avgConfidence,
            timestamp: new Date(),
            entities: [],
          };

          setMessages(prev => [...prev.slice(0, -1), aiMessage]);
          setIsGenerating(false);
          return;
        }
      } catch (fallbackError) {
        console.log('Both APIs failed, using mock data:', fallbackError);
      }
    }

    // Fallback to mock data
    await new Promise(resolve => setTimeout(resolve, 1500));

    const citations = getCitationsForQuestion(questionText);

    const answerMap: Record<string, string> = {
      '合同总金额是多少？': '根据合同第五条的约定，本合同项目的总金额为人民币500万元整（¥5,000,000.00）。这是双方经过友好协商后确定的项目总价款。',
      '谁是甲方和乙方？': '根据合同首部的记载，甲方是科技创新发展有限公司，乙方是智能解决方案股份有限公司。双方均为依法设立并有效存续的企业法人。',
      '海洋污染的主要原因是什么？': '根据研究报告，海洋污染的主要原因包括塑料废弃物和工业排放。研究数据显示，每年约有800万吨塑料废弃物进入海洋环境，对海洋生态系统造成严重影响。',
      '患者的主要诊断是什么？': '根据病历记录，患者的主要诊断是冠状动脉粥样硬化性心脏病（冠心病），同时合并有高血压病（2级，高危）。这些诊断是基于临床检查和影像学检查结果综合得出的。',
    };

    const answer = answerMap[questionText] || '抱歉，我在文档中没有找到足够的证据来回答这个问题。请尝试询问文档中明确提到的内容。';
    const confidence = citations.length > 0 ? 0.95 : 0;

    // Add AI message
    const aiMessage: Message = {
      id: `ai-${Date.now()}`,
      type: 'ai',
      content: answer,
      citations,
      confidence,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, aiMessage]);
    setIsGenerating(false);
  };

  const handleAskQuestion = () => {
    if (question.trim()) {
      generateAnswer(question);
    }
  };

  const handleExampleClick = (exampleQuestion: string) => {
    generateAnswer(exampleQuestion);
  };

  // 渲染带实体高亮的文本
  const renderHighlightedText = (content: string, entities: MatchedEntity[]) => {
    if (!entities || entities.length === 0 || viewMode === 'traditional') {
      return content;
    }

    // 1. 去重：相同位置范围的实体只保留一个
    const uniqueEntities: MatchedEntity[] = [];
    const seenRanges = new Set<string>();

    for (const entity of entities) {
      const rangeKey = `${entity.start_pos}-${entity.end_pos}`;
      if (!seenRanges.has(rangeKey)) {
        seenRanges.add(rangeKey);
        uniqueEntities.push(entity);
      }
    }

    // 2. 按起始位置排序（从前往后）
    const sortedEntities = [...uniqueEntities].sort((a, b) => a.start_pos - b.start_pos);

    // 3. 移除重叠的实体，只保留不重叠的
    const nonOverlappingEntities: MatchedEntity[] = [];
    let lastEndPos = -1;

    for (const entity of sortedEntities) {
      // 只添加不与前一个实体重叠的实体
      if (entity.start_pos >= lastEndPos) {
        nonOverlappingEntities.push(entity);
        lastEndPos = entity.end_pos;
      }
    }

    // 4. 创建文本片段数组
    const segments: { text: string; entity?: MatchedEntity }[] = [];
    let currentPos = 0;

    for (const entity of nonOverlappingEntities) {
      // 添加实体前面的普通文本
      if (entity.start_pos > currentPos) {
        segments.push({ text: content.slice(currentPos, entity.start_pos) });
      }
      // 添加实体
      segments.push({
        text: content.slice(entity.start_pos, entity.end_pos),
        entity
      });
      currentPos = entity.end_pos;
    }

    // 添加最后一个实体后面的文本
    if (currentPos < content.length) {
      segments.push({ text: content.slice(currentPos) });
    }

    return (
      <>
        {segments.map((segment, index) => {
          if (segment.entity) {
            const color = getEntityColor(segment.entity.entity_type);
            return (
              <span
                key={index}
                className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md ${color.bg} ${color.text} ${color.border} border cursor-pointer hover:opacity-80 transition-opacity`}
                onClick={() => setSelectedEntity(segment.entity!)}
                title={`${segment.entity.entity_type} (置信度: ${Math.round(segment.entity.confidence * 100)}%)`}
              >
                <Tag className="w-3 h-3" />
                {segment.text}
              </span>
            );
          }
          return <span key={index}>{segment.text}</span>;
        })}
      </>
    );
  };

  const handleCitationClick = (citation: Citation, messageId: string) => {
    setSelectedCitation(citation);
    setSelectedMessageId(messageId);
  };

  return (
    <div className="w-full">
      {/* Chat Interface - 左右两栏布局，使用 CSS Grid 确保稳定 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: '24px' }}>
        {/* 左侧：聊天窗口 */}
        <div className="floating-card h-[600px] flex flex-col overflow-hidden">
            {/* Header */}
            <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--background)]">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shadow-sm">
                    <Sparkles className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-[var(--foreground)]">
                      {isKnowledgeBaseMode ? 'AI 知识库助手' : 'AI 文档助手'}
                    </h3>
                    <div className="flex items-center gap-2 text-[0.75rem] text-[var(--foreground-muted)] tracking-tight">
                      <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                      <span>
                        {isKnowledgeBaseMode ? '知识库已就绪，随时为您服务' : '已加载文档，随时为您服务'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* 模式切换按钮 */}
                <div className="flex items-center gap-2 bg-gray-100 rounded-full p-1">
                  <button
                    onClick={() => setViewMode('traditional')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[0.75rem] font-medium transition-all ${
                      viewMode === 'traditional'
                        ? 'bg-white text-gray-900 shadow-sm'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    传统模式
                  </button>
                  <button
                    onClick={() => setViewMode('structured')}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[0.75rem] font-medium transition-all ${
                      viewMode === 'structured'
                        ? 'bg-gradient-to-r from-[var(--primary)] to-indigo-600 text-white shadow-sm'
                        : 'text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    <GitBranch className="w-3.5 h-3.5" />
                    结构化模式
                  </button>
                </div>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {messages.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center max-w-md">
                    <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
                      <BookOpen className="w-10 h-10 text-gray-400" />
                    </div>
                    <h4 className="text-[var(--foreground)] mb-2">开始对话</h4>
                    <p className="text-[0.875rem] text-[var(--foreground-muted)] tracking-tight">
                      向AI提出关于文档的问题，获得基于证据的答案
                    </p>
                  </div>
                </div>
              ) : (
                <>
                  {messages.map((message) => (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      {message.type === 'ai' && (
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center flex-shrink-0 mr-3 mt-1 shadow-sm">
                          <Sparkles className="w-4 h-4 text-white" />
                        </div>
                      )}

                      <div className={`max-w-[80%] ${message.type === 'user' ? 'ml-auto' : ''}`}>
                        {/* Message Bubble */}
                        <div
                          className={`p-4 rounded-[var(--radius-lg)] shadow-[var(--shadow-sm)] ${
                            message.type === 'user'
                              ? 'bg-gradient-to-br from-[var(--primary)] to-indigo-600 text-white'
                              : 'bg-[var(--background-elevated)] border border-[var(--border)]'
                          }`}
                        >
                          {message.type === 'ai' ? (
                            <div className="text-[0.9375rem] leading-relaxed tracking-tight text-[var(--foreground)]">
                              {viewMode === 'structured' && message.structuredContent && !isGenerating ? (
                                // 结构化卡片模式（仅在生成完成后显示）
                                <div className="space-y-3">
                                  {/* 核心答案摘要 - 更突出显示 */}
                                  <div className="p-4 bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50 rounded-xl border border-blue-200 shadow-sm">
                                    <div className="flex items-start gap-3">
                                      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center flex-shrink-0">
                                        <Zap className="w-4 h-4 text-white" />
                                      </div>
                                      <div>
                                        <p className="text-xs text-blue-600 font-semibold mb-1">核心回答</p>
                                        <p className="text-sm text-gray-900 leading-relaxed">{message.structuredContent.summary}</p>
                                      </div>
                                    </div>
                                  </div>

                                  {/* 详细要点标题 */}
                                  {message.structuredContent.items.length > 0 && (
                                    <div className="flex items-center gap-2 pt-2">
                                      <Layers className="w-4 h-4 text-gray-400" />
                                      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">详细要点</span>
                                    </div>
                                  )}

                                  {/* 条目卡片列表 */}
                                  <div className="space-y-2">
                                    {message.structuredContent.items.map((item, idx) => (
                                      <motion.div
                                        key={idx}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: idx * 0.1 }}
                                        className="p-3 bg-white rounded-lg border border-gray-200 hover:border-[var(--primary)] hover:shadow-sm transition-all"
                                      >
                                        <div className="flex items-center gap-2 mb-2">
                                          <span className="w-5 h-5 rounded-full bg-gradient-to-br from-[var(--primary)] to-indigo-600 text-white text-xs flex items-center justify-center font-medium">
                                            {idx + 1}
                                          </span>
                                          <h4 className="font-semibold text-gray-900 text-sm">{item.title}</h4>
                                        </div>
                                        {/* 高亮标签 */}
                                        {item.highlights && item.highlights.length > 0 && (
                                          <div className="flex flex-wrap gap-1.5 mb-2">
                                            {item.highlights.map((h, i) => (
                                              <span key={i} className="px-2 py-0.5 bg-amber-100 text-amber-800 text-xs rounded-full font-medium">
                                                {h}
                                              </span>
                                            ))}
                                          </div>
                                        )}
                                        {/* 详细说明 */}
                                        {item.detail && (
                                          <p className="text-gray-600 text-sm leading-relaxed">{item.detail}</p>
                                        )}
                                      </motion.div>
                                    ))}
                                  </div>

                                  {/* 结论（如果有） */}
                                  {message.structuredContent.conclusion && (
                                    <div className="p-3 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
                                      <div className="flex items-start gap-2">
                                        <CheckCircle2 className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                                        <p className="text-sm text-green-900">{message.structuredContent.conclusion}</p>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              ) : (
                                // 流式输出中或传统模式：使用 Markdown 渲染
                                <ReactMarkdown
                                  remarkPlugins={[remarkGfm]}
                                  components={{
                                    p: ({node, ...props}) => <p className="mb-2" {...props} />,
                                    ul: ({node, ...props}) => <ul className="list-disc ml-4 mb-2" {...props} />,
                                    ol: ({node, ...props}) => <ol className="list-decimal ml-4 mb-2" {...props} />,
                                    li: ({node, ...props}) => <li className="mb-1" {...props} />,
                                    strong: ({node, ...props}) => <strong className="font-semibold text-[var(--primary)]" {...props} />,
                                    code: ({node, ...props}) => <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono" {...props} />,
                                  }}
                                >
                                  {message.content}
                                </ReactMarkdown>
                              )}
                            </div>
                          ) : (
                            <p className="text-[0.9375rem] leading-relaxed tracking-tight text-white">
                              {message.content}
                            </p>
                          )}

                          {/* AI Confidence Indicator */}
                          {message.type === 'ai' && message.confidence !== undefined && message.confidence > 0 && (
                            <div className="mt-3 pt-3 border-t border-[var(--border)] flex items-center gap-2">
                              <Eye className="w-4 h-4 text-green-600" />
                              <span className="text-[0.75rem] text-[var(--foreground-muted)] tracking-tight">
                                置信度：
                              </span>
                              <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden max-w-[100px]">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${message.confidence * 100}%` }}
                                  transition={{ duration: 0.8, delay: 0.3 }}
                                  className="h-full bg-gradient-to-r from-green-500 to-emerald-600 rounded-full"
                                />
                              </div>
                              <span className="text-[0.75rem] font-semibold text-green-600">
                                {Math.round(message.confidence * 100)}%
                              </span>
                            </div>
                          )}
                        </div>

                        {/* Citation Chips - 证据引用标签 */}
                        {message.type === 'ai' && message.citations && message.citations.length > 0 && (
                          <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.3 }}
                            className="mt-3 flex flex-wrap gap-2"
                          >
                            {message.citations.map((citation, index) => (
                              <motion.button
                                key={index}
                                ref={(el) => {
                                  if (el) citationRefs.current.set(`${message.id}-${index}`, el);
                                }}
                                initial={{ opacity: 0, y: 5 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.4 + index * 0.05 }}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => handleCitationClick(citation, message.id)}
                                className={`group relative flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[0.75rem] font-medium transition-all ${
                                  selectedCitation === citation
                                    ? 'bg-amber-100 text-amber-800 border-2 border-amber-400 shadow-md'
                                    : 'bg-gradient-to-r from-blue-50 to-indigo-50 text-indigo-700 border border-indigo-200 hover:border-indigo-400 hover:shadow-sm'
                                }`}
                              >
                                <Quote className="w-3 h-3" />
                                <span>
                                  {citation.extraction_class || `段落 ${citation.page}`}
                                </span>
                                <span className={`ml-1 px-1.5 py-0.5 rounded text-[0.625rem] ${
                                  selectedCitation === citation
                                    ? 'bg-amber-200 text-amber-900'
                                    : 'bg-indigo-100 text-indigo-600'
                                }`}>
                                  {Math.round(citation.confidence * 100)}%
                                </span>
                              </motion.button>
                            ))}
                          </motion.div>
                        )}

                        {/* Timestamp */}
                        <div className="mt-2 text-[0.6875rem] text-[var(--foreground-subtle)] tracking-tight">
                          {message.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>

                      {message.type === 'user' && (
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-gray-300 to-gray-400 flex items-center justify-center flex-shrink-0 ml-3 mt-1">
                          <span className="text-[0.75rem] font-semibold text-white">你</span>
                        </div>
                      )}
                    </motion.div>
                  ))}


                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Input Area */}
            <div className="px-6 py-4 border-t border-[var(--border)] bg-[var(--background)]">
              <div className="flex gap-3">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleAskQuestion()}
                  placeholder="输入您的问题..."
                  disabled={isGenerating}
                  className="flex-1 px-4 py-3 border-2 border-[var(--border)] rounded-[var(--radius-md)] focus:border-[var(--primary)] focus:outline-none text-[var(--foreground)] bg-[var(--background-elevated)] transition-colors tracking-tight disabled:opacity-50"
                />
                <motion.button
                  onClick={handleAskQuestion}
                  disabled={!question.trim() || isGenerating}
                  className={`px-6 py-3 rounded-[var(--radius-md)] transition-all flex items-center gap-2 tracking-tight ${
                    question.trim() && !isGenerating
                      ? 'bg-gradient-to-r from-[var(--primary)] to-indigo-600 text-white shadow-[0_4px_20px_var(--primary-glow)] hover:shadow-[0_6px_25px_var(--primary-glow)]'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  }`}
                  whileHover={question.trim() && !isGenerating ? { scale: 1.02 } : {}}
                  whileTap={question.trim() && !isGenerating ? { scale: 0.98 } : {}}
                >
                  <Send className="w-5 h-5" />
                  <span>发送</span>
                </motion.button>
              </div>
            </div>
          </div>

        {/* 右侧：证据详情面板 */}
        <div className="floating-card h-[600px] flex flex-col overflow-hidden">
          {/* 如果有选中的引用，显示证据详情 */}
          {selectedCitation ? (
            <>
              {/* 头部 */}
              <div className="px-5 py-4 bg-gradient-to-r from-amber-50 to-orange-50 border-b border-amber-200 flex items-center justify-between flex-shrink-0">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center shadow-lg">
                    <FileText className="w-5 h-5 text-white" strokeWidth={2} />
                  </div>
                  <div>
                    <h4 className="text-base font-bold text-gray-900">原文证据</h4>
                    <p className="text-xs text-amber-700 mt-0.5">{selectedCitation.location}</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedCitation(null)}
                  className="w-8 h-8 rounded-full bg-amber-100 hover:bg-amber-200 flex items-center justify-center transition-colors"
                >
                  <X className="w-4 h-4 text-amber-700" strokeWidth={2} />
                </button>
              </div>

              {/* 滚动内容区 */}
              <div className="flex-1 overflow-y-auto">
                {/* 文档原文与溯源高亮 */}
                <div className="p-4 border-b border-gray-100">
                  <div className="flex items-center gap-2 mb-3">
                    <FileSearch className="w-4 h-4 text-amber-600" />
                    <span className="text-xs font-semibold text-gray-700">原文溯源</span>
                    {selectedCitation.extraction_class && (
                      <span className="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs font-medium">
                        {selectedCitation.extraction_class}
                      </span>
                    )}
                  </div>

                  {/* 原文高亮显示 - 从 attributes 或文本中提取关键词进行多色高亮 */}
                  {(() => {
                    // 从 attributes 中提取要高亮的关键词
                    const attrs = selectedCitation.attributes || {};
                    const keywords: Array<{ text: string; label?: string }> = [];

                    // 提取所有有意义的属性值作为关键词
                    const skipKeys = ['paragraph_index', 'source', 'char_interval', 'extraction_class', 'confidence', 'score'];
                    Object.entries(attrs).forEach(([key, value]) => {
                      if (!skipKeys.includes(key) && value && typeof value === 'string' && value.length >= 2 && value.length <= 50) {
                        keywords.push({ text: value, label: key });
                      }
                    });

                    // 如果有 extraction_class，也加入高亮
                    const extractionClass = selectedCitation.extraction_class || attrs.extraction_class;
                    if (extractionClass && typeof extractionClass === 'string') {
                      keywords.unshift({ text: extractionClass, label: '类型' });
                    }

                    // 如果没有从 attributes 提取到关键词，自动从文本中提取专业术语
                    const displayText = selectedCitation.text || '';
                    if (keywords.length === 0 && displayText) {
                      // 医学/生物学专业术语正则匹配
                      const patterns = [
                        // 信号通路
                        /TGF-?\s*β?\/?\s*Smad\d*/gi,
                        /PI3K\/?\s*Akt/gi,
                        /NF-?\s*κ?\s*B/gi,
                        /Wnt\/?\s*β?-?\s*catenin/gi,
                        /MAPK/gi,
                        /mTOR/gi,
                        /JAK\/?\s*STAT/gi,
                        // 细胞类型
                        /HSC|肝星状细胞/g,
                        /肝细胞/g,
                        /巨噬细胞/g,
                        /Kupffer\s*细胞/gi,
                        // 组学
                        /转录组学/g,
                        /蛋白质组学/g,
                        /代谢组学/g,
                        /基因组学/g,
                        // 中药相关
                        /中药复方/g,
                        /复方[\u4e00-\u9fa5]{2,6}/g,
                        /[\u4e00-\u9fa5]{2,4}方/g,
                        /[\u4e00-\u9fa5]{2,4}汤/g,
                        /[\u4e00-\u9fa5]{2,4}散/g,
                        /[\u4e00-\u9fa5]{2,4}丸/g,
                        // 分子/蛋白
                        /miRNA|circRNA|lncRNA/gi,
                        /Smad\d+/gi,
                        /Collagen|胶原/gi,
                        /α-SMA/gi,
                        /PPARγ/gi,
                        // 疾病
                        /肝纤维化/g,
                        /肝硬化/g,
                        /肝癌/g,
                      ];

                      const foundTerms = new Set<string>();
                      patterns.forEach((pattern, idx) => {
                        const matches = displayText.match(pattern);
                        if (matches) {
                          matches.forEach(m => {
                            if (m.length >= 2 && !foundTerms.has(m)) {
                              foundTerms.add(m);
                              keywords.push({ text: m, label: '术语' });
                            }
                          });
                        }
                      });

                      // 限制关键词数量
                      if (keywords.length > 8) {
                        keywords.length = 8;
                      }
                    }

                    return (
                      <div className="text-sm text-gray-600 leading-relaxed">
                        <div className="relative pl-3 border-l-2 border-amber-400 bg-amber-50/50 p-3 rounded-r-lg">
                          <p className="text-gray-900 whitespace-pre-wrap leading-relaxed">
                            <MultiKeywordHighlight text={displayText} keywords={keywords} />
                          </p>
                        </div>
                        {/* 关键词图例 */}
                        {keywords.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {keywords.slice(0, 8).map((kw, idx) => {
                              const color = HIGHLIGHT_COLORS[idx % HIGHLIGHT_COLORS.length];
                              return (
                                <span
                                  key={idx}
                                  className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] ${color.bg} ${color.text}`}
                                >
                                  <span className="font-medium">{kw.label || '关键词'}:</span>
                                  <span>{kw.text.length > 12 ? kw.text.substring(0, 12) + '..' : kw.text}</span>
                                </span>
                              );
                            })}
                            {keywords.length > 8 && (
                              <span className="text-[10px] text-gray-400">+{keywords.length - 8} 更多</span>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })()}

                  {/* 溯源位置信息 */}
                  <div className="mt-3 flex flex-wrap gap-2 text-xs text-gray-500">
                    {selectedCitation.char_interval && (
                      <span className="px-2 py-1 bg-amber-100 text-amber-700 rounded font-medium">
                        📍 字符位置: {selectedCitation.char_interval.start_pos} - {selectedCitation.char_interval.end_pos}
                      </span>
                    )}
                    <span className="px-2 py-1 bg-gray-100 rounded">
                      📄 段落 {selectedCitation.page}
                    </span>
                    {!originalText && (
                      <span className="px-2 py-1 bg-yellow-50 text-yellow-700 rounded text-[10px]">
                        💡 原文内容未加载，请先进行文档提取
                      </span>
                    )}
                  </div>
                </div>

                {/* 置信度和来源 */}
                <div className="p-4 border-b border-gray-100">
                  <div className="flex items-center justify-between p-3 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5 text-green-600" />
                      <span className="text-sm font-medium text-green-900">语义相关度</span>
                    </div>
                    <div className="text-lg font-bold text-green-700">
                      {Math.round(selectedCitation.confidence * 100)}%
                    </div>
                  </div>
                </div>

                {/* 实体属性（如果有） - 来自检索结果的 attributes，用不同颜色区分 */}
                {selectedCitation.attributes && Object.keys(selectedCitation.attributes).length > 0 && (
                  <div className="p-4 border-b border-gray-100">
                    <div className="flex items-center gap-2 mb-3">
                      <Info className="w-4 h-4 text-blue-600" />
                      <span className="text-xs font-semibold text-blue-900">实体属性</span>
                    </div>
                    <div className="space-y-2">
                      {Object.entries(selectedCitation.attributes)
                        .filter(([key]) => !['paragraph_index', 'source', 'char_interval', 'confidence', 'score'].includes(key))
                        .map(([key, value], idx) => {
                          const color = HIGHLIGHT_COLORS[idx % HIGHLIGHT_COLORS.length];
                          return (
                            <div
                              key={idx}
                              className={`flex items-center gap-2 p-2 rounded-lg border ${color.bg} border-opacity-50`}
                              style={{ borderColor: color.bg.replace('bg-', '').replace('-200', '-300') }}
                            >
                              <span className={`text-xs font-bold ${color.text} min-w-[60px]`}>{key}:</span>
                              <span className={`text-xs ${color.text} font-medium`}>{String(value)}</span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                )}

                {/* 知识图谱可视化 - 当有 attributes 或 extraction_class 时显示 */}
                {(() => {
                  const attrs = selectedCitation.attributes || {};
                  const extractionClass = selectedCitation.extraction_class || attrs.extraction_class;
                  const hasValidAttrs = Object.keys(attrs).filter(k =>
                    !['paragraph_index', 'source', 'char_interval'].includes(k)
                  ).length > 0;

                  if (!hasValidAttrs && !extractionClass) return null;

                  return (
                    <div className="p-4 border-b border-gray-100">
                      <div className="flex items-center gap-2 mb-3">
                        <Network className="w-4 h-4 text-purple-600" />
                        <span className="text-xs font-semibold text-purple-900">知识图谱可视化</span>
                      </div>
                      <KnowledgeGraphVisualization
                        attributes={attrs}
                        extractionClass={extractionClass}
                        contentText={selectedCitation.text}
                      />
                    </div>
                  );
                })()}

                {/* 相关实体 - 从当前消息的匹配实体中获取 */}
                <div className="p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Tag className="w-4 h-4 text-blue-600" />
                    <span className="text-xs font-semibold text-blue-900">相关实体</span>
                    <span className="ml-auto text-xs text-gray-400">
                      共 {currentMessageEntities.length} 个
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {currentMessageEntities.length > 0 ? (
                      currentMessageEntities.slice(0, 15).map((entity, index) => {
                        const color = getEntityColor(entity.entity_type);
                        return (
                          <span
                            key={index}
                            className={`px-2 py-1 rounded text-xs font-medium ${color.bg} ${color.text} ${color.border} border`}
                            title={entity.entity_type}
                          >
                            {entity.text}
                          </span>
                        );
                      })
                    ) : (
                      <p className="text-xs text-gray-500 italic">暂无相关实体</p>
                    )}
                    {currentMessageEntities.length > 15 && (
                      <span className="px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-600">
                        +{currentMessageEntities.length - 15} 更多
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </>
          ) : (
            /* 默认状态：显示示例问题和透明度承诺 */
            <div className="flex-1 overflow-y-auto">
              {/* Example Questions */}
              {messages.length === 0 && (
                <div className="p-4 border-b border-gray-100">
                  <h4 className="text-[var(--foreground)] mb-3 flex items-center gap-2 text-sm font-semibold">
                    <Zap className="w-4 h-4 text-[var(--primary)]" />
                    <span>试试这些问题</span>
                  </h4>
                  <div className="space-y-2">
                    {questions.map((q, index) => (
                      <motion.button
                        key={index}
                        onClick={() => handleExampleClick(q)}
                        className="w-full text-left px-3 py-2 rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 hover:shadow-sm transition-all text-sm"
                        whileHover={{ scale: 1.01, x: 2 }}
                        whileTap={{ scale: 0.99 }}
                      >
                        "{q}"
                      </motion.button>
                    ))}
                  </div>
                </div>
              )}

              {/* Entity Detail Panel */}
              {selectedEntity && viewMode === 'structured' && (
                <div className="p-4 border-b border-gray-100">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Tag className="w-4 h-4 text-indigo-700" />
                      <span className="text-xs font-semibold text-indigo-900">实体详情</span>
                    </div>
                    <button
                      onClick={() => setSelectedEntity(null)}
                      className="w-6 h-6 rounded-full hover:bg-indigo-100 flex items-center justify-center"
                    >
                      <X className="w-3 h-3 text-indigo-700" />
                    </button>
                  </div>
                  <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-200 mb-3">
                    <p className="font-semibold text-indigo-900">{selectedEntity.text}</p>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-500">类型</span>
                    <span className={`px-2 py-0.5 rounded-full ${getEntityColor(selectedEntity.entity_type).bg} ${getEntityColor(selectedEntity.entity_type).text}`}>
                      {selectedEntity.entity_type}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs mt-2">
                    <span className="text-gray-500">置信度</span>
                    <span className="font-semibold text-indigo-600">{Math.round(selectedEntity.confidence * 100)}%</span>
                  </div>
                </div>
              )}

              {/* Transparency Card */}
              <div className="p-4">
                <h4 className="text-purple-900 mb-3 flex items-center gap-2 text-sm font-semibold">
                  <ShieldCheck className="w-4 h-4 text-purple-600" />
                  <span>透明度承诺</span>
                </h4>
                <div className="space-y-2 text-xs text-purple-800">
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-purple-600" />
                    <span>每个答案都附带可点击的引用芯片</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-purple-600" />
                    <span>点击芯片即可查看原文证据预览</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-purple-600" />
                    <span>显示置信度评分，帮助您判断可靠性</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-purple-600" />
                    <span>AI 正在"阅读"文档，而非杜撰内容</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
