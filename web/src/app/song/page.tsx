import { Suspense } from "react";
import { Loader } from "@/components/InputSection";
import SongView from "./SongView";

// /song/?id=<track id>&key=<artist:title>&d=<diversity>. The id is in the query, not the path, because a
// static export cannot pre-build a page per song.
export default function SongPage() {
  return (
    <Suspense fallback={<Loader text="Loading song..." />}>
      <SongView />
    </Suspense>
  );
}
