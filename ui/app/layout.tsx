import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Training ROI Predictor (Causal Uplift)",
  description: "X-learner CATE for L&D interventions.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
