import { Tooltip as T } from "radix-ui";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

export const TooltipProvider = T.Provider;

export function Tooltip({
  content,
  children,
  side = "bottom",
  className,
}: {
  content: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}) {
  if (!content) return <>{children}</>;
  return (
    <T.Root>
      <T.Trigger asChild>{children}</T.Trigger>
      <T.Portal>
        <T.Content
          side={side}
          sideOffset={6}
          collisionPadding={8}
          className={cn(
            "z-50 max-w-72 rounded-md bg-ink px-2.5 py-1.5 text-xs leading-relaxed text-bg shadow-lg",
            className,
          )}
        >
          {content}
        </T.Content>
      </T.Portal>
    </T.Root>
  );
}
