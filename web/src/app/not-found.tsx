import Link from "next/link";

export default function NotFound() {
  return (
    <div className="panel" style={{ textAlign: "center" }}>
      <h3>Page not found</h3>
      <p className="panel-lead">There is nothing at this address.</p>
      <Link href="/" className="btn btn-play">Back to Music Discovery</Link>
    </div>
  );
}
