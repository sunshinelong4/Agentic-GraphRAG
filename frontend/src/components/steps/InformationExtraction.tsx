import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChevronRight, ChevronLeft, Sparkles, Loader2, CheckCircle2, Eye, Zap, AlertTriangle, Database, Save } from 'lucide-react';
import { mockDocuments } from '../../data/mockDocuments';
import { TeachingBanner } from '../TeachingMode';
import { apiService, ScenarioType, ExtractionItem, ExtractionResponse } from '../../services/api';

interface InformationExtractionProps {
  onNext: () => void;
  onPrevious: () => void;
  selectedDocument: string | null;
}

interface ExtractedItem {
  id: string;
  type: string;
  value: string;
  role: string;
  highlight: string;
  pass: 1 | 2;
  confidence: number;
  attributes?: Record<string, string>;
}

interface HighlightSpan {
  text: string;
  itemId: string;
  type: string;
  color: string;
}

type ExtractionStage = 'idle' | 'pass1' | 'pass2' | 'complete' | 'loading' | 'error';

export function InformationExtraction({ onNext, onPrevious, selectedDocument }: InformationExtractionProps) {
  const [stage, setStage] = useState<ExtractionStage>('idle');
  const [activeCardId, setActiveCardId] = useState<string | null>(null);
  const [extractedItems, setExtractedItems] = useState<ExtractedItem[]>([]);
  const [currentExplanation, setCurrentExplanation] = useState('');

  // 粘贴文本相关状态
  const [pastedText, setPastedText] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [processingTime, setProcessingTime] = useState<number | null>(null);

  // 知识库存储状态
  const [isSavingToKnowledge, setIsSavingToKnowledge] = useState(false);
  const [saveResult, setSaveResult] = useState<{ success: boolean; message: string } | null>(null);
  const [extractionResponse, setExtractionResponse] = useState<ExtractionResponse | null>(null);

  const documentRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const highlightRefs = useRef<Map<string, HTMLSpanElement>>(new Map());

  // 加载粘贴的文本 - 组件挂载和 selectedDocument 变化时都重新加载
  useEffect(() => {
    if (selectedDocument === 'pasted-text' || selectedDocument === 'pdf-parsed') {
      const text = sessionStorage.getItem('pastedText') || '';
      setPastedText(text);
      console.log('[InformationExtraction] Loaded pastedText from sessionStorage:', text.substring(0, 50) + '...');
    }
  }, [selectedDocument]);

  // 确保组件挂载时也检查 sessionStorage
  useEffect(() => {
    if ((selectedDocument === 'pasted-text' || selectedDocument === 'pdf-parsed') && !pastedText) {
      const text = sessionStorage.getItem('pastedText') || '';
      if (text) {
        setPastedText(text);
      }
    }
  }, []);

  if (!selectedDocument) {
    return <div>请先选择一个文档</div>;
  }

  // 判断是粘贴文本还是示例文档
  const isPastedText = selectedDocument === 'pasted-text' || selectedDocument === 'pdf-parsed';
  const doc = !isPastedText ? mockDocuments[selectedDocument as keyof typeof mockDocuments] : null;

  // 获取文档内容 - 优先使用 state，如果为空则从 sessionStorage 读取
  const getDocumentContent = () => {
    if (!isPastedText) {
      return doc?.content || '';
    }
    // 对于粘贴文本，确保从 sessionStorage 获取最新值
    if (pastedText && pastedText.length > 0) {
      return pastedText;
    }
    // 如果 state 为空，尝试从 sessionStorage 读取
    const storedText = sessionStorage.getItem('pastedText') || '';
    return storedText;
  };

  const documentContent = getDocumentContent();

  // 调试输出
  console.log('[InformationExtraction] documentContent length:', documentContent?.length, 'pastedText length:', pastedText?.length);

  // Convert entities to extracted items (仅用于示例文档)
  const allItems: ExtractedItem[] = doc ? doc.extracted.entities.map((entity, index) => ({
    id: `item-${index}`,
    type: entity.type,
    value: entity.value,
    role: entity.role,
    highlight: entity.highlight,
    pass: index < doc.extracted.entities.length / 2 ? 1 : 2,
    confidence: 85 + Math.random() * 14,
  })) : [];

  const pass1Items = allItems.filter(item => item.pass === 1);
  const pass2Items = allItems.filter(item => item.pass === 2);

  // 示例文档的模拟提取
  useEffect(() => {
    if (isPastedText) return; // 粘贴文本使用真实 API
    if (stage === 'idle') return;

    if (stage === 'pass1') {
      setCurrentExplanation('正在识别文档中的核心信息，包括关键人物、组织、日期和金额...');

      pass1Items.forEach((item, index) => {
        setTimeout(() => {
          setExtractedItems(prev => [...prev, item]);
        }, index * 400);
      });

      setTimeout(() => {
        setStage('pass2');
      }, pass1Items.length * 400 + 500);
    }

    if (stage === 'pass2') {
      setCurrentExplanation('正在补充提取额外的细节信息，完善数据完整性...');

      pass2Items.forEach((item, index) => {
        setTimeout(() => {
          setExtractedItems(prev => [...prev, item]);
        }, index * 400);
      });

      setTimeout(() => {
        setStage('complete');
        setCurrentExplanation('提取完成！已从文档中识别出所有关键信息。');
      }, pass2Items.length * 400 + 500);
    }
  }, [stage, isPastedText]);

  // 开始提取
  const startExtraction = async () => {
    setExtractedItems([]);
    setErrorMessage('');

    if (isPastedText) {
      // 粘贴文本：调用真实后端 API
      setStage('loading');
      setCurrentExplanation('正在调用 DeepSeek API 进行智能提取...');

      try {
        const scenario = (sessionStorage.getItem('selectedScenario') || 'news') as ScenarioType;
        // 直接从 sessionStorage 读取文本，确保获取最新值
        const textToExtract = sessionStorage.getItem('pastedText') || pastedText;

        console.log('[InformationExtraction] Starting extraction:');
        console.log('  - scenario:', scenario);
        console.log('  - textToExtract length:', textToExtract?.length);
        console.log('  - textToExtract preview:', textToExtract?.substring(0, 100));

        if (!textToExtract.trim()) {
          setStage('error');
          setErrorMessage('文本内容为空，请返回上一步重新输入');
          return;
        }

        const result = await apiService.extract({
          text: textToExtract,
          scenario,
          use_cache: true,
        });

        console.log('[InformationExtraction] API result:', result);

        if (result.success && result.extractions.length > 0) {
          // 保存原始响应用于后续存入知识库
          setExtractionResponse(result);

          // 转换 API 返回结果为 ExtractedItem 格式
          const items: ExtractedItem[] = result.extractions.map((ext, index) => ({
            id: `item-${index}`,
            type: ext.extraction_class,
            value: ext.extraction_text,
            role: ext.attributes?.['类型'] || ext.attributes?.['职位'] || ext.extraction_class,
            highlight: ext.extraction_text,
            pass: index < result.extractions.length / 2 ? 1 : 2,
            confidence: 85 + Math.random() * 14,
            attributes: ext.attributes,
          }));

          setProcessingTime(result.processing_time || null);

          // 保存提取结果到 sessionStorage，供知识图谱使用
          sessionStorage.setItem('extractedItems', JSON.stringify(items));

          // 逐个显示提取结果
          items.forEach((item, index) => {
            setTimeout(() => {
              setExtractedItems(prev => [...prev, item]);
            }, index * 200);
          });

          setTimeout(() => {
            setStage('complete');
            setCurrentExplanation('提取完成！已从文本中识别出所有关键信息。');
          }, items.length * 200 + 300);
        } else {
          setStage('error');
          setErrorMessage(result.error || '未能提取到任何信息');
        }
      } catch (error) {
        console.error('[InformationExtraction] Error:', error);
        setStage('error');
        if (error instanceof Error) {
          setErrorMessage(error.message);
        } else if (typeof error === 'object' && error !== null) {
          setErrorMessage(JSON.stringify(error, null, 2));
        } else {
          setErrorMessage('网络错误，请检查后端服务');
        }
      }
    } else {
      // 示例文档：使用模拟提取
      setStage('pass1');
    }
  };

  // 保存提取结果到知识库
  const saveToKnowledgeBase = async () => {
    if (!extractionResponse || !extractionResponse.extractions.length) {
      setSaveResult({ success: false, message: '没有可保存的提取结果' });
      return;
    }

    setIsSavingToKnowledge(true);
    setSaveResult(null);

    try {
      // 获取文档标题
      const docTitle = sessionStorage.getItem('documentTitle') ||
                       sessionStorage.getItem('pdfFileName') ||
                       `提取文档-${new Date().toLocaleString()}`;

      // 调用 API 存储提取结果
      const result = await apiService.addExtractions({
        doc_title: docTitle,
        extractions: extractionResponse.extractions.map(ext => ({
          extraction_class: ext.extraction_class,
          extraction_text: ext.extraction_text,
          char_interval: ext.char_interval,
          attributes: ext.attributes || {},
        })),
      });

      if (result.success) {
        setSaveResult({
          success: true,
          message: `成功保存 ${result.chunk_count} 条知识到知识库`,
        });
      } else {
        setSaveResult({
          success: false,
          message: result.error || '保存失败',
        });
      }
    } catch (error) {
      setSaveResult({
        success: false,
        message: error instanceof Error ? error.message : '网络错误',
      });
    } finally {
      setIsSavingToKnowledge(false);
    }
  };

  const getTypeColor = (type: string): { bg: string; border: string; text: string; highlight: string; highlightColor: string } => {
    const colorMap: Record<string, { bg: string; border: string; text: string; highlight: string; highlightColor: string }> = {
      // 基础实体类型
      '人物': { bg: 'from-blue-50 to-blue-100', border: 'border-blue-300', text: 'text-blue-700', highlight: 'bg-blue-200', highlightColor: 'rgba(191, 219, 254, 0.8)' },
      '组织': { bg: 'from-purple-50 to-purple-100', border: 'border-purple-300', text: 'text-purple-700', highlight: 'bg-purple-200', highlightColor: 'rgba(233, 213, 255, 0.8)' },
      '机构': { bg: 'from-purple-50 to-purple-100', border: 'border-purple-300', text: 'text-purple-700', highlight: 'bg-purple-200', highlightColor: 'rgba(233, 213, 255, 0.8)' },
      '日期': { bg: 'from-green-50 to-green-100', border: 'border-green-300', text: 'text-green-700', highlight: 'bg-green-200', highlightColor: 'rgba(187, 247, 208, 0.8)' },
      '时间': { bg: 'from-green-50 to-green-100', border: 'border-green-300', text: 'text-green-700', highlight: 'bg-green-200', highlightColor: 'rgba(187, 247, 208, 0.8)' },
      '金额': { bg: 'from-amber-50 to-amber-100', border: 'border-amber-300', text: 'text-amber-700', highlight: 'bg-amber-200', highlightColor: 'rgba(253, 230, 138, 0.8)' },
      '地址': { bg: 'from-pink-50 to-pink-100', border: 'border-pink-300', text: 'text-pink-700', highlight: 'bg-pink-200', highlightColor: 'rgba(251, 207, 232, 0.8)' },
      '地点': { bg: 'from-pink-50 to-pink-100', border: 'border-pink-300', text: 'text-pink-700', highlight: 'bg-pink-200', highlightColor: 'rgba(251, 207, 232, 0.8)' },
      '数量': { bg: 'from-indigo-50 to-indigo-100', border: 'border-indigo-300', text: 'text-indigo-700', highlight: 'bg-indigo-200', highlightColor: 'rgba(199, 210, 254, 0.8)' },
      '时长': { bg: 'from-rose-50 to-rose-100', border: 'border-rose-300', text: 'text-rose-700', highlight: 'bg-rose-200', highlightColor: 'rgba(254, 205, 211, 0.8)' },
      '产品': { bg: 'from-sky-50 to-sky-100', border: 'border-sky-300', text: 'text-sky-700', highlight: 'bg-sky-200', highlightColor: 'rgba(186, 230, 253, 0.8)' },
      '事件': { bg: 'from-rose-50 to-rose-100', border: 'border-rose-300', text: 'text-rose-700', highlight: 'bg-rose-200', highlightColor: 'rgba(254, 205, 211, 0.8)' },

      // 医疗/生物相关
      '疾病': { bg: 'from-red-50 to-red-100', border: 'border-red-300', text: 'text-red-700', highlight: 'bg-red-200', highlightColor: 'rgba(254, 202, 202, 0.8)' },
      '病症': { bg: 'from-red-50 to-red-100', border: 'border-red-300', text: 'text-red-700', highlight: 'bg-red-200', highlightColor: 'rgba(254, 202, 202, 0.8)' },
      '药物': { bg: 'from-cyan-50 to-cyan-100', border: 'border-cyan-300', text: 'text-cyan-700', highlight: 'bg-cyan-200', highlightColor: 'rgba(165, 243, 252, 0.8)' },
      '中药': { bg: 'from-emerald-50 to-emerald-100', border: 'border-emerald-300', text: 'text-emerald-700', highlight: 'bg-emerald-200', highlightColor: 'rgba(167, 243, 208, 0.8)' },
      '测量值': { bg: 'from-violet-50 to-violet-100', border: 'border-violet-300', text: 'text-violet-700', highlight: 'bg-violet-200', highlightColor: 'rgba(221, 214, 254, 0.8)' },
      '百分比': { bg: 'from-orange-50 to-orange-100', border: 'border-orange-300', text: 'text-orange-700', highlight: 'bg-orange-200', highlightColor: 'rgba(254, 215, 170, 0.8)' },
      '物种': { bg: 'from-teal-50 to-teal-100', border: 'border-teal-300', text: 'text-teal-700', highlight: 'bg-teal-200', highlightColor: 'rgba(153, 246, 228, 0.8)' },
      '手术': { bg: 'from-fuchsia-50 to-fuchsia-100', border: 'border-fuchsia-300', text: 'text-fuchsia-700', highlight: 'bg-fuchsia-200', highlightColor: 'rgba(245, 208, 254, 0.8)' },
      '生物过程': { bg: 'from-lime-50 to-lime-100', border: 'border-lime-300', text: 'text-lime-700', highlight: 'bg-lime-200', highlightColor: 'rgba(217, 249, 157, 0.8)' },
      '信号通路': { bg: 'from-yellow-50 to-yellow-100', border: 'border-yellow-400', text: 'text-yellow-700', highlight: 'bg-yellow-200', highlightColor: 'rgba(254, 240, 138, 0.8)' },
      '组学类型': { bg: 'from-indigo-50 to-indigo-100', border: 'border-indigo-300', text: 'text-indigo-700', highlight: 'bg-indigo-200', highlightColor: 'rgba(199, 210, 254, 0.8)' },
      '细胞类型': { bg: 'from-pink-50 to-pink-100', border: 'border-pink-300', text: 'text-pink-700', highlight: 'bg-pink-200', highlightColor: 'rgba(251, 207, 232, 0.8)' },
      '基因': { bg: 'from-violet-50 to-violet-100', border: 'border-violet-300', text: 'text-violet-700', highlight: 'bg-violet-200', highlightColor: 'rgba(221, 214, 254, 0.8)' },
      '蛋白质': { bg: 'from-blue-50 to-blue-100', border: 'border-blue-300', text: 'text-blue-700', highlight: 'bg-blue-200', highlightColor: 'rgba(191, 219, 254, 0.8)' },
      '研究方法': { bg: 'from-slate-50 to-slate-100', border: 'border-slate-300', text: 'text-slate-700', highlight: 'bg-slate-200', highlightColor: 'rgba(226, 232, 240, 0.8)' },
      '实验技术': { bg: 'from-gray-50 to-gray-100', border: 'border-gray-300', text: 'text-gray-700', highlight: 'bg-gray-300', highlightColor: 'rgba(209, 213, 219, 0.8)' },
    };
    return colorMap[type] || { bg: 'from-gray-50 to-gray-100', border: 'border-gray-300', text: 'text-gray-700', highlight: 'bg-gray-200', highlightColor: 'rgba(229, 231, 235, 0.8)' };
  };

  const getTypeIcon = (type: string) => {
    const icons: Record<string, string> = {
      '组织': '🏢', '机构': '🏢', '人物': '👤', '地址': '📍', '地点': '📍',
      '日期': '📅', '时间': '📅', '金额': '💰', '数量': '🔢', '时长': '⏱️',
      '病症': '🩺', '药物': '💊', '测量值': '📊', '百分比': '📈', '物种': '🐟',
      '手术': '⚕️', '产品': '📦', '事件': '📋',
    };
    return icons[type] || '📌';
  };

  // Parse document content with highlights
  const renderDocumentWithHighlights = () => {
    const content = documentContent;
    const highlights: Array<{ start: number; end: number; itemId: string; type: string; text: string }> = [];

    // 调试: 检查内容和提取项
    console.log('[renderDocumentWithHighlights] content length:', content?.length);
    console.log('[renderDocumentWithHighlights] extractedItems count:', extractedItems.length);
    if (extractedItems.length > 0) {
      console.log('[renderDocumentWithHighlights] first item:', extractedItems[0]);
    }

    if (!content || content.length === 0) {
      console.warn('[renderDocumentWithHighlights] No content to highlight!');
      return <span>{content}</span>;
    }

    // Find all highlight positions - 使用多种匹配策略
    extractedItems.forEach(item => {
      // 优先使用 value 字段（实体名称），因为 highlight 可能是长文本片段
      const searchTexts = [
        item.value,  // 实体值（如 "肝纤维化"）
        item.highlight,  // 高亮文本（可能是长片段）
      ].filter(t => t && t.length >= 2);

      let foundIndex = -1;
      let matchLength = 0;
      let matchedText = '';

      // 尝试每个候选文本
      for (const searchText of searchTexts) {
        if (foundIndex !== -1) break;

        // 策略1: 精确匹配
        const idx = content.indexOf(searchText);
        if (idx !== -1) {
          foundIndex = idx;
          matchLength = searchText.length;
          matchedText = searchText;
          console.log(`[highlight] Found "${searchText}" at index ${idx}`);
          break;
        }

        // 策略2: 如果文本较长，尝试匹配前20个字符
        if (searchText.length > 20) {
          const partialText = searchText.substring(0, 20);
          const partialIdx = content.indexOf(partialText);
          if (partialIdx !== -1) {
            foundIndex = partialIdx;
            matchLength = partialText.length;
            matchedText = partialText;
            console.log(`[highlight] Found partial "${partialText}" at index ${partialIdx}`);
            break;
          }
        }
      }

      if (foundIndex === -1) {
        // 详细调试：检查字符编码差异
        const valueChars = item.value?.split('').map(c => c.charCodeAt(0)).join(',');
        console.warn(`[highlight] NOT FOUND: value="${item.value}" (chars: ${valueChars})`);
        // 尝试在内容中搜索相似文本
        const searchVal = item.value || '';
        if (searchVal.length >= 2) {
          const firstTwoChars = searchVal.substring(0, 2);
          let searchPos = 0;
          const occurrences: string[] = [];
          while ((searchPos = content.indexOf(firstTwoChars, searchPos)) !== -1 && occurrences.length < 3) {
            occurrences.push(`pos ${searchPos}: "${content.substring(searchPos, searchPos + 10)}..."`);
            searchPos++;
          }
          if (occurrences.length > 0) {
            console.log(`[highlight] Similar text found for "${firstTwoChars}": ${occurrences.join(', ')}`);
          }
        }
      }

      if (foundIndex !== -1 && matchLength > 0) {
        // 检查是否与已有高亮重叠
        const isOverlapping = highlights.some(h =>
          (foundIndex >= h.start && foundIndex < h.end) ||
          (foundIndex + matchLength > h.start && foundIndex + matchLength <= h.end) ||
          (foundIndex <= h.start && foundIndex + matchLength >= h.end)
        );
        if (!isOverlapping) {
          highlights.push({
            start: foundIndex,
            end: foundIndex + matchLength,
            itemId: item.id,
            type: item.type,
            text: matchedText,
          });
        }
      }
    });

    console.log('[renderDocumentWithHighlights] highlights found:', highlights.length);

    // 如果没有高亮，直接返回原文
    if (highlights.length === 0) {
      console.log('[renderDocumentWithHighlights] No highlights, returning plain text');
      return <span>{content}</span>;
    }

    // Sort by position
    highlights.sort((a, b) => a.start - b.start);
    console.log('[renderDocumentWithHighlights] First 3 highlights:', highlights.slice(0, 3));

    // Build JSX with highlights
    const elements: React.ReactNode[] = [];
    let lastIndex = 0;

    highlights.forEach((hl, i) => {
      // Add text before highlight
      if (hl.start > lastIndex) {
        elements.push(
          <span key={`text-${i}`}>{content.substring(lastIndex, hl.start)}</span>
        );
      }

      // Add highlighted span
      const colors = getTypeColor(hl.type);
      const isActive = activeCardId === hl.itemId;

      elements.push(
        <span
          key={`hl-${i}`}
          ref={(el) => {
            if (el) highlightRefs.current.set(hl.itemId, el);
          }}
          style={{ backgroundColor: colors.highlightColor }}
          className={`px-1 py-0.5 rounded transition-all duration-200 cursor-pointer font-medium ${
            isActive ? 'ring-2 ring-[var(--primary)] shadow-md scale-[1.02]' : 'hover:shadow-sm'
          }`}
          onClick={() => setActiveCardId(hl.itemId)}
          title={`${hl.type}: ${hl.text.substring(0, 50)}${hl.text.length > 50 ? '...' : ''}`}
        >
          {content.substring(hl.start, hl.end)}
        </span>
      );

      lastIndex = hl.end;
    });

    // Add remaining text
    if (lastIndex < content.length) {
      elements.push(
        <span key="text-end">{content.substring(lastIndex)}</span>
      );
    }

    console.log('[renderDocumentWithHighlights] Total elements created:', elements.length);
    return elements;
  };

  // Calculate Bezier curve path
  const calculateCurvePath = (itemId: string): string | null => {
    const cardEl = cardRefs.current.get(itemId);
    const highlightEl = highlightRefs.current.get(itemId);
    const containerEl = documentRef.current;

    if (!cardEl || !highlightEl || !containerEl) return null;

    const containerRect = containerEl.getBoundingClientRect();
    const highlightRect = highlightEl.getBoundingClientRect();
    const cardRect = cardEl.getBoundingClientRect();

    // Start point (right edge of highlight in document)
    const startX = highlightRect.right - containerRect.left;
    const startY = highlightRect.top + highlightRect.height / 2 - containerRect.top;

    // End point (left edge of card)
    const endX = cardRect.left - containerRect.left;
    const endY = cardRect.top + cardRect.height / 2 - containerRect.top;

    // Control points for elegant Bezier curve
    const midX = (startX + endX) / 2;
    const cp1X = midX;
    const cp1Y = startY;
    const cp2X = midX;
    const cp2Y = endY;

    return `M ${startX} ${startY} C ${cp1X} ${cp1Y}, ${cp2X} ${cp2Y}, ${endX} ${endY}`;
  };

  return (
    <div className="space-y-6">
      {/* Teaching Mode Banner */}
      <TeachingBanner
        title="这一步在做什么？"
        description={isPastedText
          ? "AI 正在调用 DeepSeek 大模型分析您粘贴的文本，识别并提取关键信息。每个提取结果都可追溯到原文位置。"
          : "AI 正在阅读文档，就像一个细心的助手用荧光笔标记重要信息。它会识别人名、公司名、日期、金额等关键数据点，然后将它们整理成结构化的格式。"
        }
        tips={[
          '观察左侧文档中的彩色高亮 - 每种颜色代表不同类型的信息',
          '点击右侧的数据卡片，优雅的曲线会显示它在原文中的确切位置',
          '每个提取的数据都能追溯到原文位置，确保准确性',
        ]}
        comparison={isPastedText
          ? "提取结果来自真实的 LLM 处理，不是预设数据。"
          : "这个过程替代了人工逐行阅读30多页文档并手动整理到表格中，通常需要2-3小时的工作量。"
        }
      />

      {/* Step Indicator */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-[var(--primary-light)] to-purple-50 border border-[var(--primary)]/20 rounded-full shadow-[var(--shadow-sm)] mb-4">
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-[var(--primary)] to-purple-600 text-white flex items-center justify-center shadow-sm">
            <span className="text-[0.875rem] font-semibold">2</span>
          </div>
          <span className="text-[0.875rem] font-medium text-[var(--primary)] tracking-tight">
            第 2 步，共 4 步 – 提取结构化信息
          </span>
        </div>
        <h2 className="text-[var(--foreground)] mb-3">AI 信息提取过程</h2>
        <p className="text-[var(--foreground-muted)] tracking-tight">
          {isPastedText ? '使用 LangExtract + DeepSeek 从文本中提取关键信息' : '观察 AI 如何从非结构化文本中识别和提取关键信息'}
        </p>
      </div>

      {/* Error State */}
      {stage === 'error' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-50 border-2 border-red-200 rounded-xl p-6"
        >
          <div className="flex items-start gap-4">
            <AlertTriangle className="w-8 h-8 text-red-500 flex-shrink-0" />
            <div className="flex-1">
              <h4 className="text-red-900 mb-2">提取失败</h4>
              <p className="text-sm text-red-700 mb-4">{errorMessage}</p>
              <div className="flex gap-3">
                <button
                  onClick={() => setStage('idle')}
                  className="px-4 py-2 bg-white border border-red-300 text-red-700 rounded-lg hover:bg-red-50"
                >
                  重试
                </button>
                <button
                  onClick={onPrevious}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                >
                  返回修改
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Start Button */}
      {stage === 'idle' && (
        <div className="flex justify-center py-12">
          <motion.button
            onClick={startExtraction}
            className="group relative flex items-center gap-3 px-12 py-5 bg-gradient-to-r from-[var(--primary)] to-purple-600 text-white rounded-full shadow-[0_8px_30px_var(--primary-glow)] overflow-hidden"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.98 }}
          >
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent"
              animate={{ x: ['-100%', '100%'] }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            />
            <Sparkles className="w-6 h-6 relative z-10" />
            <span className="text-[1.125rem] font-semibold tracking-tight relative z-10">开始智能提取</span>
          </motion.button>
        </div>
      )}

      {/* Loading State */}
      {stage === 'loading' && (
        <div className="flex flex-col items-center justify-center py-16">
          <Loader2 className="w-12 h-12 text-[var(--primary)] animate-spin mb-4" />
          <h3 className="text-[var(--foreground)] mb-2">正在调用 DeepSeek API...</h3>
          <p className="text-[var(--foreground-muted)] text-sm">{currentExplanation}</p>
        </div>
      )}

      {/* Split View Interface */}
      {(stage === 'pass1' || stage === 'pass2' || stage === 'complete' || (stage === 'loading' && extractedItems.length > 0)) && (
        <div ref={documentRef} className="relative">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-2 gap-8 h-[700px]"
          >
            {/* LEFT PANEL - Source Document */}
            <div className="floating-card flex flex-col overflow-hidden">
              {/* Header */}
              <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--background)]">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--foreground)] mb-1 flex items-center gap-2">
                      <Eye className="w-5 h-5 text-[var(--primary)]" />
                      源文档
                    </h3>
                    <p className="text-[0.8125rem] text-[var(--foreground-muted)] tracking-tight">
                      彩色高亮显示已提取的信息
                    </p>
                  </div>
                  {stage !== 'idle' && stage !== 'error' && (
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--primary-light)] rounded-full">
                      {stage !== 'complete' && <Loader2 className="w-4 h-4 text-[var(--primary)] animate-spin" />}
                      <span className="text-[0.75rem] font-medium text-[var(--primary)] tracking-tight">
                        {stage === 'pass1' && '第一轮'}
                        {stage === 'pass2' && '第二轮'}
                        {stage === 'loading' && '提取中'}
                        {stage === 'complete' && '完成'}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Document Content */}
              <div className="flex-1 overflow-y-auto p-6">
                <div className="text-[0.8125rem] leading-relaxed text-[var(--foreground-muted)] whitespace-pre-wrap">
                  {renderDocumentWithHighlights()}
                </div>
              </div>
            </div>

            {/* RIGHT PANEL - Insight Cards */}
            <div className="floating-card flex flex-col overflow-hidden">
              {/* Header */}
              <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--background)]">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-[var(--foreground)] mb-1 flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-purple-600" />
                      提取洞察
                    </h3>
                    <p className="text-[0.8125rem] text-[var(--foreground-muted)] tracking-tight">
                      点击卡片查看源证据
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-[1.25rem] font-bold text-[var(--primary)] tracking-tight">
                      {extractedItems.length}
                    </div>
                    <div className="text-[0.6875rem] text-[var(--foreground-muted)] tracking-tight">
                      数据项
                    </div>
                  </div>
                </div>
              </div>

              {/* Cards List */}
              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                <AnimatePresence>
                  {extractedItems.map((item, index) => {
                    const colors = getTypeColor(item.type);
                    const isActive = activeCardId === item.id;

                    return (
                      <motion.div
                        key={item.id}
                        ref={(el) => {
                          if (el) cardRefs.current.set(item.id, el);
                        }}
                        initial={{ opacity: 0, x: 50, scale: 0.9 }}
                        animate={{
                          opacity: 1,
                          x: 0,
                          scale: isActive ? 1.03 : 1,
                        }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        transition={{
                          delay: index * 0.05,
                          type: 'spring',
                          stiffness: 300,
                          damping: 25,
                        }}
                        onClick={() => setActiveCardId(isActive ? null : item.id)}
                        className={`relative cursor-pointer p-4 rounded-[var(--radius-lg)] border-2 transition-all duration-200 ${
                          isActive
                            ? 'border-[var(--primary)] shadow-[var(--shadow-lg)] bg-gradient-to-br from-white to-blue-50'
                            : `bg-gradient-to-br ${colors.bg} ${colors.border} hover:shadow-[var(--shadow-md)]`
                        }`}
                      >
                        {/* Glow effect when active */}
                        {isActive && (
                          <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="absolute inset-0 rounded-[var(--radius-lg)] bg-gradient-to-br from-[var(--primary)]/10 to-purple-500/10 pointer-events-none"
                          />
                        )}

                        <div className="relative z-10 flex items-start gap-3">
                          {/* Icon */}
                          <div className={`w-11 h-11 rounded-[var(--radius-md)] flex items-center justify-center shadow-sm ${
                            isActive ? 'bg-gradient-to-br from-[var(--primary)] to-purple-600' : 'bg-white border border-[var(--border)]'
                          }`}>
                            <span className="text-lg">
                              {isActive ? '✨' : getTypeIcon(item.type)}
                            </span>
                          </div>

                          {/* Content */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className={`text-[0.6875rem] font-medium tracking-tight ${
                                isActive ? 'text-[var(--primary)]' : colors.text
                              }`}>
                                {item.type}
                              </span>
                              {item.pass === 2 && !isPastedText && (
                                <span className="px-1.5 py-0.5 bg-purple-100 text-purple-700 text-[0.625rem] font-semibold rounded tracking-tight">
                                  补充
                                </span>
                              )}
                              {item.attributes && Object.keys(item.attributes).length > 0 && (
                                <span className="px-1.5 py-0.5 bg-blue-100 text-blue-700 text-[0.625rem] font-semibold rounded tracking-tight">
                                  {Object.keys(item.attributes).length} 属性
                                </span>
                              )}
                            </div>
                            <div className={`text-[0.9375rem] font-semibold mb-1 tracking-tight break-words ${
                              isActive ? 'text-[var(--foreground)]' : colors.text.replace('700', '900')
                            }`}>
                              {item.value}
                            </div>
                            <div className="text-[0.75rem] text-[var(--foreground-muted)] tracking-tight">
                              {item.role}
                            </div>

                            {/* Attributes or Confidence indicator */}
                            {isActive && (
                              <motion.div
                                initial={{ opacity: 0, y: -5 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="mt-3 pt-3 border-t border-[var(--border)]"
                              >
                                {item.attributes && Object.keys(item.attributes).length > 0 ? (
                                  <div>
                                    <div className="text-[0.6875rem] text-[var(--foreground-muted)] mb-2">属性</div>
                                    <div className="flex flex-wrap gap-2">
                                      {Object.entries(item.attributes).map(([key, value]) => (
                                        <span
                                          key={key}
                                          className="px-2 py-1 bg-gray-100 rounded text-[0.75rem] text-gray-700"
                                        >
                                          {key}: {value}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                ) : (
                                  <>
                                    <div className="flex items-center justify-between mb-1">
                                      <span className="text-[0.6875rem] text-[var(--foreground-muted)] tracking-tight">
                                        置信度
                                      </span>
                                      <span className="text-[0.75rem] font-semibold text-green-600">
                                        {item.confidence.toFixed(1)}%
                                      </span>
                                    </div>
                                    <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                      <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${item.confidence}%` }}
                                        transition={{ duration: 0.5, delay: 0.2 }}
                                        className="h-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full"
                                      />
                                    </div>
                                  </>
                                )}
                              </motion.div>
                            )}
                          </div>

                          {/* Active indicator */}
                          {isActive && (
                            <motion.div
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              className="w-6 h-6 rounded-full bg-[var(--primary)] flex items-center justify-center flex-shrink-0 shadow-sm"
                            >
                              <Zap className="w-3.5 h-3.5 text-white" />
                            </motion.div>
                          )}
                        </div>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </div>
          </motion.div>

          {/* Bezier Curve Connections - SVG Overlay */}
          <svg
            className="absolute inset-0 pointer-events-none"
            style={{ width: '100%', height: '100%' }}
          >
            <defs>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {activeCardId && calculateCurvePath(activeCardId) && (
              <>
                {/* Glow layer */}
                <motion.path
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 0.3 }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                  d={calculateCurvePath(activeCardId) || ''}
                  fill="none"
                  stroke="var(--primary)"
                  strokeWidth="8"
                  filter="url(#glow)"
                />

                {/* Main line */}
                <motion.path
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                  d={calculateCurvePath(activeCardId) || ''}
                  fill="none"
                  stroke="var(--primary)"
                  strokeWidth="2"
                  strokeDasharray="0"
                />

                {/* Animated flowing particles */}
                {[0, 0.33, 0.66].map((offset) => (
                  <motion.circle
                    key={offset}
                    r="4"
                    fill="var(--primary)"
                    initial={{ offsetDistance: '0%' }}
                    animate={{ offsetDistance: '100%' }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: 'linear',
                      delay: offset,
                    }}
                    style={{
                      offsetPath: `path("${calculateCurvePath(activeCardId)}")`,
                    }}
                  />
                ))}
              </>
            )}
          </svg>
        </div>
      )}

      {/* Completion Summary */}
      {stage === 'complete' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="floating-card p-6 bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-300"
        >
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-[var(--radius-md)] bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center flex-shrink-0 shadow-md">
              <CheckCircle2 className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h4 className="text-green-900 mb-2">提取完成！</h4>
              <p className="text-[0.875rem] text-green-800 leading-relaxed tracking-tight mb-4">
                成功从文档中提取了 <strong>{extractedItems.length}</strong> 个数据项，涵盖 <strong>{new Set(extractedItems.map(i => i.type)).size}</strong> 种信息类型。每个提取结果都可以追溯到原文位置，确保数据准确性和可验证性。
                {processingTime && (
                  <span className="ml-2 text-green-600">(处理耗时: {processingTime.toFixed(2)}秒)</span>
                )}
              </p>
              <div className="flex items-center gap-2 text-[0.8125rem] text-green-700 mb-4">
                <Sparkles className="w-4 h-4" />
                <span className="tracking-tight">点击右侧的数据卡片，查看优雅的连接线指向源证据</span>
              </div>

              {/* 保存到知识库按钮 - 仅对粘贴文本或PDF显示 */}
              {/* 调试信息 */}
              {console.log('[InformationExtraction] isPastedText:', isPastedText, 'extractionResponse:', !!extractionResponse, 'selectedDocument:', selectedDocument)}
              {isPastedText && extractionResponse && (
                <div className="pt-4 border-t border-green-200">
                  <div className="flex items-center gap-3">
                    <motion.button
                      onClick={saveToKnowledgeBase}
                      disabled={isSavingToKnowledge || saveResult?.success}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        saveResult?.success
                          ? 'bg-green-600 text-white cursor-default'
                          : isSavingToKnowledge
                          ? 'bg-gray-200 text-gray-500 cursor-wait'
                          : 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:shadow-lg hover:scale-105'
                      }`}
                      whileHover={!isSavingToKnowledge && !saveResult?.success ? { scale: 1.02 } : {}}
                      whileTap={!isSavingToKnowledge && !saveResult?.success ? { scale: 0.98 } : {}}
                    >
                      {isSavingToKnowledge ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>保存中...</span>
                        </>
                      ) : saveResult?.success ? (
                        <>
                          <CheckCircle2 className="w-4 h-4" />
                          <span>已保存</span>
                        </>
                      ) : (
                        <>
                          <Database className="w-4 h-4" />
                          <span>保存到知识库</span>
                        </>
                      )}
                    </motion.button>

                    {/* 保存结果提示 */}
                    {saveResult && (
                      <motion.span
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        className={`text-sm ${saveResult.success ? 'text-green-700' : 'text-red-600'}`}
                      >
                        {saveResult.message}
                      </motion.span>
                    )}
                  </div>
                  <p className="text-xs text-green-600 mt-2">
                    💡 保存后可在"智能问答"步骤中基于这些知识进行问答，并查看溯源信息
                  </p>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Navigation */}
      <div className="flex justify-between items-center pt-4">
        <motion.button
          onClick={onPrevious}
          className="flex items-center gap-2 px-6 py-3 bg-[var(--background-elevated)] border-2 border-[var(--border)] rounded-[var(--radius-md)] text-[var(--foreground)] hover:border-[var(--primary)] hover:shadow-[var(--shadow-md)] transition-all"
          whileHover={{ scale: 1.02, x: -2 }}
          whileTap={{ scale: 0.98 }}
        >
          <ChevronLeft className="w-5 h-5" />
          <span className="tracking-tight">上一步</span>
        </motion.button>

        <motion.button
          onClick={onNext}
          disabled={stage !== 'complete'}
          className={`flex items-center gap-2 px-8 py-3 rounded-[var(--radius-md)] transition-all tracking-tight ${
            stage === 'complete'
              ? 'bg-gradient-to-r from-[var(--primary)] to-purple-600 text-white shadow-[0_4px_20px_var(--primary-glow)] hover:shadow-[0_8px_30px_var(--primary-glow)]'
              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          }`}
          whileHover={stage === 'complete' ? { scale: 1.02, x: 2 } : {}}
          whileTap={stage === 'complete' ? { scale: 0.98 } : {}}
        >
          <span>继续到知识图谱</span>
          <ChevronRight className="w-5 h-5" />
        </motion.button>
      </div>
    </div>
  );
}
