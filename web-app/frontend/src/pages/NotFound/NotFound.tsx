import { Link } from 'react-router-dom';

/** Catch-all 404 page for unmatched routes */
const NotFound: React.FC = () => (
  <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
    <h1 className="text-6xl font-bold text-accent-blue mb-4">404</h1>
    <p className="text-xl text-gray-300 mb-8">Page not found</p>
    <Link
      to="/"
      className="px-6 py-3 bg-accent-blue text-white rounded-lg hover:bg-blue-600 transition-colors"
    >
      Back to Home
    </Link>
  </div>
);

export default NotFound;
