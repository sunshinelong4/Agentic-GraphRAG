import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowRight, Sparkles, FileText, Database, Users, TrendingUp, CheckCircle, Zap } from 'lucide-react';

interface LandingProps {
  onGetStarted: () => void;
}

export function Landing({ onGetStarted }: LandingProps) {
  const [scanProgress, setScanProgress] = useState(0);
  
  // Continuous scanning animation
  useEffect(() => {
    const interval = setInterval(() => {
      setScanProgress((prev) => (prev >= 100 ? 0 : prev + 2));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen relative overflow-hidden bg-[var(--background)]">
      {/* Ambient background gradients */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute top-0 -left-1/4 w-[600px] h-[600px] bg-gradient-to-br from-blue-200/30 to-indigo-200/30 rounded-full blur-3xl"
          animate={{
            x: [0, 50, 0],
            y: [0, 30, 0],
            scale: [1, 1.1, 1],
          }}
          transition={{ duration: 20, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className="absolute bottom-0 -right-1/4 w-[500px] h-[500px] bg-gradient-to-br from-purple-200/30 to-pink-200/30 rounded-full blur-3xl"
          animate={{
            x: [0, -30, 0],
            y: [0, -50, 0],
            scale: [1, 1.2, 1],
          }}
          transition={{ duration: 15, repeat: Infinity, ease: 'easeInOut' }}
        />
      </div>

      <div className="relative z-10 container mx-auto px-6 py-20 max-w-[1400px]">
        {/* Header Badge */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex justify-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-[var(--primary-light)] to-blue-50 border border-[var(--primary)]/20 rounded-full shadow-[var(--shadow-sm)]">
            <Sparkles className="w-4 h-4 text-[var(--primary)]" />
            <span className="text-[0.875rem] font-medium text-[var(--primary)] tracking-tight">
              赋范空间 · 《大模型Agent智能体开发实战》课程产品
            </span>
          </div>
        </motion.div>

        {/* Hero Section - Split Screen with 3D Transition */}
        <div className="grid lg:grid-cols-[1fr_auto_1fr] gap-12 items-center mb-24">
          {/* LEFT SIDE - Chaotic Documents */}
          <motion.div
            initial={{ opacity: 0, x: -100 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: [0.4, 0, 0.2, 1] }}
            className="relative"
          >
            {/* Ethereal PDF Background Layer - Behind the stack */}
            <div className="absolute inset-0 -inset-x-32 -inset-y-32 pointer-events-none overflow-visible z-0">
              {/* Multiple floating ethereal PDF documents */}
              {[0, 1, 2, 3, 4, 5].map((index) => (
                <motion.div
                  key={`bg-${index}`}
                  className="absolute bg-gradient-to-br from-blue-50/40 to-indigo-50/40 backdrop-blur-md rounded-3xl border border-blue-100/25 shadow-lg"
                  style={{
                    width: `${320 + index * 45}px`,
                    height: `${420 + index * 60}px`,
                    left: `${-35 + index * 22}%`,
                    top: `${-25 + index * 15}%`,
                  }}
                  animate={{
                    y: [0, -35, 0],
                    rotate: [-12 + index * 5, -8 + index * 5, -12 + index * 5],
                    opacity: [0.12, 0.22, 0.12],
                  }}
                  transition={{
                    duration: 5 + index * 1.5,
                    repeat: Infinity,
                    ease: 'easeInOut',
                    delay: index * 0.35,
                  }}
                >
                  {/* Simulated PDF text lines - subtle and elegant */}
                  <div className="p-8 space-y-3 opacity-40">
                    <div className="h-2 bg-blue-300/50 rounded-full w-3/5 mb-4"></div>
                    {Array.from({ length: 12 }).map((_, i) => (
                      <div
                        key={i}
                        className="h-1 bg-gray-300/40 rounded-full"
                        style={{ width: `${70 + Math.random() * 25}%` }}
                      ></div>
                    ))}
                    <div className="h-1.5 bg-blue-300/50 rounded-full w-1/2 mt-4"></div>
                    {Array.from({ length: 8 }).map((_, i) => (
                      <div
                        key={`sub-${i}`}
                        className="h-1 bg-gray-300/40 rounded-full"
                        style={{ width: `${65 + Math.random() * 30}%` }}
                      ></div>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>

            <div className="relative perspective-[1000px] z-10">
              {/* Stacked messy documents */}
              {[0, 1, 2, 3, 4].map((index) => (
                <motion.div
                  key={index}
                  className="absolute inset-0 bg-white rounded-[var(--radius-xl)] border border-[var(--border)] shadow-[var(--shadow-lg)] overflow-hidden"
                  style={{
                    transformOrigin: 'center center',
                  }}
                  initial={{
                    rotate: -15 + index * 7,
                    x: -30 + index * 15,
                    y: -20 + index * 10,
                    scale: 0.95 + index * 0.01,
                  }}
                  animate={{
                    rotate: -15 + index * 7 + Math.sin(index) * 2,
                    x: -30 + index * 15,
                    y: -20 + index * 10,
                    scale: 0.95 + index * 0.01,
                  }}
                  transition={{
                    duration: 3 + index * 0.5,
                    repeat: Infinity,
                    repeatType: 'reverse',
                    ease: 'easeInOut',
                  }}
                >
                  {/* Document content - dense and overwhelming */}
                  <div className="p-6 h-[500px] overflow-hidden">
                    <div className="space-y-2 text-[0.65rem] leading-tight text-gray-600">
                      <div className="h-2 bg-gray-300 rounded w-3/4 mb-3"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-full"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-11/12"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-full"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-10/12"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-full"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-9/12"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-full"></div>
                      <div className="h-2 bg-gray-300 rounded w-2/3 my-3"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-full"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-11/12"></div>
                      <div className="h-1.5 bg-gray-200 rounded w-full"></div>
                      {Array.from({ length: 20 }).map((_, i) => (
                        <div key={i} className="h-1.5 bg-gray-200 rounded" style={{ width: `${70 + Math.random() * 30}%` }}></div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              ))}

              {/* Chaos label */}
              <div className="absolute -top-6 -left-6 z-10">
                <div className="bg-red-50 border-2 border-red-300 rounded-[var(--radius-md)] px-4 py-2 shadow-[var(--shadow-md)]">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                    <span className="text-[0.75rem] font-semibold text-red-700 tracking-tight">混乱难读</span>
                  </div>
                  <p className="text-[0.625rem] text-red-600 mt-0.5">32 页 • 15,847 字</p>
                </div>
              </div>
            </div>
          </motion.div>

          {/* CENTER - Glowing Scanning Beam */}
          <div className="relative hidden lg:block w-32 h-[500px]">
            {/* Vertical beam container */}
            <div className="absolute inset-0 flex items-center justify-center">
              {/* Animated scanning beam */}
              <motion.div
                className="relative w-1 h-full bg-gradient-to-b from-transparent via-[var(--primary)] to-transparent opacity-30"
                animate={{
                  opacity: [0.2, 0.5, 0.2],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeInOut',
                }}
              />
              
              {/* Scanning light effect */}
              <motion.div
                className="absolute w-24 h-48 bg-gradient-to-b from-transparent via-[var(--primary)]/40 to-transparent rounded-full blur-2xl"
                animate={{
                  y: [-250, 250],
                  opacity: [0, 1, 0],
                }}
                transition={{
                  duration: 3,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              />

              {/* Glowing orb in center */}
              <motion.div
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 rounded-full bg-gradient-to-br from-[var(--primary)] to-[var(--info)] shadow-[0_0_60px_var(--primary-glow)] flex items-center justify-center"
                animate={{
                  scale: [1, 1.2, 1],
                  rotate: [0, 360],
                }}
                transition={{
                  scale: { duration: 2, repeat: Infinity, ease: 'easeInOut' },
                  rotate: { duration: 8, repeat: Infinity, ease: 'linear' },
                }}
              >
                <Zap className="w-8 h-8 text-white" />
              </motion.div>

              {/* Particle effects */}
              {Array.from({ length: 12 }).map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-1.5 h-1.5 bg-[var(--primary)] rounded-full"
                  style={{
                    left: '50%',
                    top: `${(i / 12) * 100}%`,
                  }}
                  animate={{
                    x: [0, Math.random() * 40 - 20, 40],
                    opacity: [0, 1, 0],
                    scale: [0, 1, 0],
                  }}
                  transition={{
                    duration: 2,
                    delay: i * 0.1,
                    repeat: Infinity,
                    ease: 'easeOut',
                  }}
                />
              ))}
            </div>
          </div>

          {/* RIGHT SIDE - Organized Colorful Data Cards */}
          <motion.div
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.3, ease: [0.4, 0, 0.2, 1] }}
            className="relative"
          >
            <div className="space-y-4">
              {/* Floating data cards with stagger animation */}
              {[
                { icon: '👤', label: '人物', value: '王建国', color: 'blue', delay: 0.4 },
                { icon: '🏢', label: '组织', value: '科技创新发展有限公司', color: 'purple', delay: 0.5 },
                { icon: '💰', label: '金额', value: '¥500万元', color: 'amber', delay: 0.6 },
                { icon: '📅', label: '日期', value: '2024年12月24日', color: 'green', delay: 0.7 },
                { icon: '📍', label: '地点', value: '北京市朝阳区', color: 'pink', delay: 0.8 },
              ].map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: 50, y: 20 }}
                  animate={{ opacity: 1, x: 0, y: 0 }}
                  transition={{ duration: 0.5, delay: item.delay }}
                  whileHover={{ scale: 1.05, y: -4 }}
                  className={`floating-card p-4 bg-gradient-to-br ${
                    item.color === 'blue' ? 'from-blue-50 to-blue-100 border-blue-200' :
                    item.color === 'purple' ? 'from-purple-50 to-purple-100 border-purple-200' :
                    item.color === 'amber' ? 'from-amber-50 to-amber-100 border-amber-200' :
                    item.color === 'green' ? 'from-green-50 to-green-100 border-green-200' :
                    'from-pink-50 to-pink-100 border-pink-200'
                  } border-2`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-12 h-12 rounded-[var(--radius-md)] ${
                      item.color === 'blue' ? 'bg-blue-200' :
                      item.color === 'purple' ? 'bg-purple-200' :
                      item.color === 'amber' ? 'bg-amber-200' :
                      item.color === 'green' ? 'bg-green-200' :
                      'bg-pink-200'
                    } flex items-center justify-center text-xl shadow-sm`}>
                      {item.icon}
                    </div>
                    <div className="flex-1">
                      <div className={`text-[0.6875rem] font-medium tracking-tight ${
                        item.color === 'blue' ? 'text-blue-600' :
                        item.color === 'purple' ? 'text-purple-600' :
                        item.color === 'amber' ? 'text-amber-600' :
                        item.color === 'green' ? 'text-green-600' :
                        'text-pink-600'
                      }`}>
                        {item.label}
                      </div>
                      <div className={`text-[0.9375rem] font-semibold tracking-tight mt-0.5 ${
                        item.color === 'blue' ? 'text-blue-900' :
                        item.color === 'purple' ? 'text-purple-900' :
                        item.color === 'amber' ? 'text-amber-900' :
                        item.color === 'green' ? 'text-green-900' :
                        'text-pink-900'
                      }`}>
                        {item.value}
                      </div>
                    </div>
                    <CheckCircle className={`w-5 h-5 ${
                      item.color === 'blue' ? 'text-blue-500' :
                      item.color === 'purple' ? 'text-purple-500' :
                      item.color === 'amber' ? 'text-amber-500' :
                      item.color === 'green' ? 'text-green-500' :
                      'text-pink-500'
                    }`} />
                  </div>
                </motion.div>
              ))}

              {/* Knowledge label */}
              <div className="absolute -top-6 -right-6 z-10">
                <div className="bg-green-50 border-2 border-green-300 rounded-[var(--radius-md)] px-4 py-2 shadow-[var(--shadow-md)]">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-600" />
                    <span className="text-[0.75rem] font-semibold text-green-700 tracking-tight">结构清晰</span>
                  </div>
                  <p className="text-[0.625rem] text-green-600 mt-0.5">5 类数据 • 100% 可追溯</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Headline & CTA Section */}
        <div className="text-center max-w-4xl mx-auto space-y-8">
          {/* Main Headline */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.9 }}
          >
            <h1 className="text-[3.5rem] leading-[1.1] tracking-tight mb-6">
              <span className="text-[var(--foreground)]">使用 AI，让繁杂文档井然有序</span>
            </h1>
            <p className="text-[1.25rem] text-[var(--foreground-muted)] leading-relaxed max-w-2xl mx-auto tracking-tight">
              可视化拆解 Document Agent 运行原理，从零看懂非结构化数据处理逻辑。
            </p>
          </motion.div>

          {/* CTA Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.1 }}
          >
            <motion.button
              onClick={onGetStarted}
              className="group relative inline-flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-[var(--primary)] to-[var(--info)] text-white rounded-full shadow-[0_8px_30px_var(--primary-glow)] overflow-hidden"
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              {/* Glowing background animation */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent"
                animate={{
                  x: ['-100%', '100%'],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              />
              
              <Sparkles className="w-6 h-6 relative z-10" />
              <span className="text-[1.125rem] font-semibold tracking-tight relative z-10">
                体验互动演示
              </span>
              <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform relative z-10" />
            </motion.button>
            
            <p className="text-[0.875rem] text-[var(--foreground-muted)] mt-4 tracking-tight">
              ✓ 体系教学 &nbsp;·&nbsp;  ✓ 本地部署 &nbsp;·&nbsp; ✓ 企业落地
            </p>
          </motion.div>

          {/* Feature Highlights */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.3 }}
            className="grid md:grid-cols-3 gap-6 pt-12"
          >
            {[
              { icon: FileText, title: '自动提取', desc: '识别关键实体和数据' },
              { icon: Database, title: '知识图谱', desc: '可视化关系网络' },
              { icon: Sparkles, title: '智能问答', desc: '基于证据的回答' },
            ].map((feature, index) => (
              <motion.div
                key={index}
                whileHover={{ y: -4 }}
                className="floating-card p-6 text-center"
              >
                <div className="w-14 h-14 mx-auto mb-4 rounded-[var(--radius-md)] bg-gradient-to-br from-[var(--primary-light)] to-blue-100 flex items-center justify-center">
                  <feature.icon className="w-7 h-7 text-[var(--primary)]" />
                </div>
                <h3 className="text-[var(--foreground)] mb-2">{feature.title}</h3>
                <p className="text-[0.875rem] text-[var(--foreground-muted)] tracking-tight">{feature.desc}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </div>
  );
}