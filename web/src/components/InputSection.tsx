import Recorder from "./Recorder";
import SearchBox from "./SearchBox";

export default function InputSection({ initialQuery }: { initialQuery?: string }) {
  return (
    <div className="input-section">
      <SearchBox key={initialQuery} initialQuery={initialQuery} />
      <div className="divider">or identify sound</div>
      <Recorder />
    </div>
  );
}

export function Loader({ text }: { text: string }) {
  return (
    <div className="loader" role="status">
      <div className="spinner" />
      <div>{text || "Loading..."}</div>
    </div>
  );
}
