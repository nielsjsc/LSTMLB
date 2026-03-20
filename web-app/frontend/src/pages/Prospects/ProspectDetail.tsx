import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  getProspectDetail,
  ProspectDetail as ProspectDetailType,
} from '../../services/api';

/**
 * ProspectDetail now redirects to the unified player page.
 *
 * If the prospect has an `mlbam_id` (most do), we redirect to
 * `/players/{mlbam_id}` which handles both MLB and prospect-only data.
 * If no mlbam_id exists we show a simple fallback message.
 */
export default function ProspectDetailPage() {
  const { prospectId } = useParams<{ prospectId: string }>();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!prospectId) return;
    const id = Number(prospectId);
    setLoading(true);
    getProspectDetail(id)
      .then((p: ProspectDetailType) => {
        // Prefer mlb_info.mlb_id (set when prospect has MLB stats), else mlbam_id
        const targetId = p.mlb_info?.mlb_id ?? p.mlbam_id;
        if (targetId) {
          navigate(`/players/${targetId}`, { replace: true });
        } else {
          setError('No linked player profile found for this prospect.');
          setLoading(false);
        }
      })
      .catch((e) => {
        setError(e.message || 'Prospect not found');
        setLoading(false);
      });
  }, [prospectId, navigate]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-12 text-center">
      <h2 className="text-lg font-bold text-gray-800 mb-2">Prospect</h2>
      <p className="text-gray-400 text-sm mb-4">{error}</p>
      <Link to="/prospects" className="text-blue-400 hover:text-blue-300 text-sm">
        Back to Prospects
      </Link>
    </div>
  );
}
