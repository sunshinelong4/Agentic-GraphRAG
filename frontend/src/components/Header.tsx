import React from 'react';
import { Brain, Play, X } from 'lucide-react';
import { TeachingModeToggle } from './TeachingMode';

interface HeaderProps {
  onToggleInteractive?: () => void;
  isInteractive?: boolean;
}

export function Header({ onToggleInteractive, isInteractive }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 glass border-b border-[var(--border)]">
      <div className="container mx-auto px-6 py-3 max-w-[1400px]">
        <div className="flex items-center justify-between">
          <div
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => window.location.href = '/'}
            title="返回主页"
          >
            <div className="w-9 h-9 bg-gradient-to-br from-[var(--primary)] to-[var(--info)] rounded-[14px] flex items-center justify-center shadow-lg shadow-[var(--primary-glow)]">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-[1rem] font-semibold text-[var(--foreground)] tracking-tight">
                垂直领域 Agentic-GraphRAG 检索引擎
              </h1>
              <p className="text-[0.75rem] text-[var(--foreground-muted)] mt-0.5 tracking-tight">
                {isInteractive ? '交互式教学模式' : '智能问答模式'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {onToggleInteractive && (
              <button
                onClick={onToggleInteractive}
                className={`flex items-center gap-2 px-4 py-2 rounded-[var(--radius-md)] text-[0.875rem] font-medium transition-all hover:scale-105 ${
                  isInteractive
                    ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white hover:shadow-lg'
                    : 'bg-gradient-to-r from-[var(--primary)] to-indigo-600 text-white hover:shadow-lg hover:shadow-[var(--primary-glow)]'
                }`}
              >
                {isInteractive ? (
                  <>
                    <X className="w-4 h-4" />
                    退出教学
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    交互式教学
                  </>
                )}
              </button>
            )}
            <TeachingModeToggle />
          </div>
        </div>
      </div>
    </header>
  );
}