import React, { createContext, useContext, useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { GraduationCap, X, Info } from 'lucide-react';

interface TeachingModeContextType {
  isEnabled: boolean;
  toggleTeachingMode: () => void;
}

const TeachingModeContext = createContext<TeachingModeContextType>({
  isEnabled: false,
  toggleTeachingMode: () => {},
});

export const useTeachingMode = () => useContext(TeachingModeContext);

interface TeachingModeProviderProps {
  children: React.ReactNode;
}

export function TeachingModeProvider({ children }: TeachingModeProviderProps) {
  const [isEnabled, setIsEnabled] = useState(false);

  const toggleTeachingMode = () => {
    setIsEnabled(!isEnabled);
  };

  return (
    <TeachingModeContext.Provider value={{ isEnabled, toggleTeachingMode }}>
      {children}
    </TeachingModeContext.Provider>
  );
}

interface TeachingModeToggleProps {
  className?: string;
}

export function TeachingModeToggle({ className = '' }: TeachingModeToggleProps) {
  const { isEnabled, toggleTeachingMode } = useTeachingMode();

  return (
    <motion.button
      onClick={toggleTeachingMode}
      className={`flex items-center gap-3 px-4 py-2.5 rounded-[var(--radius-md)] transition-all ${
        isEnabled 
          ? 'bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-200 shadow-[var(--shadow-sm)]' 
          : 'bg-[var(--background-elevated)] border border-[var(--border)] hover:border-amber-200'
      } ${className}`}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="relative">
        <GraduationCap className={`w-[1.125rem] h-[1.125rem] transition-colors ${
          isEnabled ? 'text-amber-600' : 'text-[var(--foreground-muted)]'
        }`} />
        {isEnabled && (
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-amber-500 rounded-full"
          />
        )}
      </div>
      <div className="text-left">
        <div className={`text-[0.8125rem] font-medium tracking-tight transition-colors ${
          isEnabled ? 'text-amber-900' : 'text-[var(--foreground)]'
        }`}>
          教学模式
        </div>
        <div className="text-[0.6875rem] text-[var(--foreground-muted)] tracking-tight">
          {isEnabled ? '显示详细解释' : '点击了解更多'}
        </div>
      </div>
      <div className={`ml-1 relative w-10 h-5 rounded-full transition-all ${
        isEnabled ? 'bg-amber-400' : 'bg-[var(--foreground-subtle)]'
      }`}>
        <motion.div
          className="absolute top-0.5 w-4 h-4 bg-white rounded-full shadow-sm"
          animate={{ left: isEnabled ? '22px' : '2px' }}
          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
        />
      </div>
    </motion.button>
  );
}

interface TeachingTooltipProps {
  title: string;
  description: string;
  comparison?: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
  children?: React.ReactNode;
}

export function TeachingTooltip({ 
  title, 
  description, 
  comparison,
  position = 'top',
  children 
}: TeachingTooltipProps) {
  const { isEnabled } = useTeachingMode();
  const [isHovered, setIsHovered] = useState(false);

  if (!isEnabled) return <>{children}</>;

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };

  return (
    <div 
      className="relative inline-block"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {children || (
        <motion.div
          className="inline-flex items-center justify-center w-6 h-6 bg-gradient-to-br from-amber-400 to-orange-500 text-white rounded-full cursor-help shadow-lg"
          whileHover={{ scale: 1.1 }}
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <Info className="w-3.5 h-3.5" />
        </motion.div>
      )}
      
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className={`absolute z-50 ${positionClasses[position]} w-80`}
          >
            <div className="bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-amber-300 rounded-xl p-4 shadow-2xl">
              <div className="flex items-start gap-2 mb-2">
                <GraduationCap className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <h4 className="text-amber-900 text-sm mb-1">{title}</h4>
                  <p className="text-xs text-amber-800 leading-relaxed">
                    {description}
                  </p>
                </div>
              </div>
              {comparison && (
                <div className="mt-3 pt-3 border-t border-amber-200">
                  <div className="flex items-start gap-2">
                    <span className="text-lg">⏱️</span>
                    <p className="text-xs text-amber-900 italic">
                      {comparison}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface TeachingBannerProps {
  title: string;
  description: string;
  tips?: string[];
  comparison?: string;
}

export function TeachingBanner({ title, description, tips, comparison }: TeachingBannerProps) {
  const { isEnabled } = useTeachingMode();
  const [isExpanded, setIsExpanded] = useState(true);

  if (!isEnabled) return null;

  return (
    <AnimatePresence>
      {isExpanded && (
        <motion.div
          initial={{ opacity: 0, y: -20, height: 0 }}
          animate={{ opacity: 1, y: 0, height: 'auto' }}
          exit={{ opacity: 0, y: -20, height: 0 }}
          className="bg-gradient-to-r from-amber-100 via-orange-100 to-yellow-100 border-2 border-amber-300 rounded-xl p-5 shadow-lg mb-6 relative overflow-hidden"
        >
          {/* Decorative background pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-2 right-2 text-6xl">📚</div>
            <div className="absolute bottom-2 left-2 text-4xl">✨</div>
          </div>

          <button
            onClick={() => setIsExpanded(false)}
            className="absolute top-3 right-3 p-1 hover:bg-amber-200 rounded-full transition-all"
          >
            <X className="w-4 h-4 text-amber-700" />
          </button>

          <div className="relative">
            <div className="flex items-start gap-3 mb-3">
              <div className="p-2 bg-amber-400 rounded-lg">
                <GraduationCap className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1">
                <h3 className="text-amber-900 mb-2">{title}</h3>
                <p className="text-sm text-amber-800 leading-relaxed">
                  {description}
                </p>
              </div>
            </div>

            {tips && tips.length > 0 && (
              <div className="ml-11 space-y-2 mb-3">
                {tips.map((tip, index) => (
                  <div key={index} className="flex items-start gap-2">
                    <span className="text-amber-600 mt-1">💡</span>
                    <span className="text-sm text-amber-900">{tip}</span>
                  </div>
                ))}
              </div>
            )}

            {comparison && (
              <div className="ml-11 mt-3 p-3 bg-white/60 rounded-lg border border-amber-200">
                <div className="flex items-center gap-2">
                  <span className="text-xl">⏱️</span>
                  <div>
                    <div className="text-xs text-amber-700 mb-1">对比人工工作</div>
                    <div className="text-sm text-amber-900">{comparison}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

interface TeachingCardProps {
  icon?: string;
  title: string;
  description: string;
  examples?: string[];
}

export function TeachingCard({ icon = '📖', title, description, examples }: TeachingCardProps) {
  const { isEnabled } = useTeachingMode();

  if (!isEnabled) return null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-xl p-4"
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl">{icon}</span>
        <div className="flex-1">
          <h4 className="text-blue-900 text-sm mb-2">{title}</h4>
          <p className="text-xs text-blue-800 leading-relaxed mb-2">
            {description}
          </p>
          {examples && examples.length > 0 && (
            <div className="space-y-1">
              {examples.map((example, index) => (
                <div key={index} className="flex items-start gap-2 text-xs text-blue-700">
                  <span className="text-blue-400">→</span>
                  <span>{example}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}