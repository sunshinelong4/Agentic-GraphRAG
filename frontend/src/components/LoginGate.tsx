import React, { useEffect, useState } from 'react';
import { apiService, AuthUser } from '../services/api';

export function LoginGate({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    apiService.currentUser().then(setUser).catch(() => apiService.clearSession()).finally(() => setLoading(false));
    const expired = () => { setUser(null); setError('登录已过期，请重新登录。'); };
    window.addEventListener('auth-expired', expired);
    return () => window.removeEventListener('auth-expired', expired);
  }, []);

  if (loading) return <div className="min-h-screen flex items-center justify-center">正在检查登录状态…</div>;
  if (user) return <>
    <div className="flex items-center justify-end gap-4 px-6 py-2 bg-slate-900 text-white text-sm">
      <span>{user.username} · {user.role === 'admin' ? '管理员' : '查询用户'}</span>
      <span>共享知识库</span>
      <button onClick={async () => {
        try { await apiService.logout(); setUser(null); setPassword(''); }
        catch (e) { window.alert(e instanceof Error ? e.message : '退出失败，请重试'); }
      }}>退出登录</button>
    </div>
    {children}
  </>;

  return <div className="min-h-screen flex items-center justify-center bg-slate-50 px-6">
    <form className="w-full max-w-md rounded-2xl bg-white p-8 shadow-lg space-y-5" onSubmit={async event => {
      event.preventDefault(); setSubmitting(true); setError('');
      try { const result = await apiService.login(username, password); setUser(result.user); setPassword(''); }
      catch (e) { setError(e instanceof Error ? e.message : '登录失败'); }
      finally { setSubmitting(false); }
    }}>
      <h1 className="text-2xl font-bold text-slate-900">Agentic-GraphRAG</h1>
      <p className="text-sm text-slate-500">登录共享知识库。管理员维护文档，查询用户检索和问答。</p>
      <label className="block">用户名<input className="mt-2 w-full rounded-lg border p-3" autoComplete="username" required value={username} onChange={e => setUsername(e.target.value)} /></label>
      <label className="block">密码<input className="mt-2 w-full rounded-lg border p-3" type="password" autoComplete="current-password" required value={password} onChange={e => setPassword(e.target.value)} /></label>
      {error && <p role="alert" className="text-red-600 text-sm">{error}</p>}
      <button disabled={submitting} className="w-full rounded-lg bg-blue-600 p-3 text-white disabled:opacity-50">{submitting ? '登录中…' : '登录'}</button>
      <p className="text-xs text-slate-500">账号由管理员创建。</p>
    </form>
  </div>;
}
