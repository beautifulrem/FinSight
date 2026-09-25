import { cva, type VariantProps } from "class-variance-authority";
import type { ComponentProps } from "react";

import { cn } from "@/lib/cn";

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg text-sm font-medium whitespace-nowrap transition-[background-color,color,box-shadow,transform] duration-150 disabled:pointer-events-none disabled:opacity-45 active:scale-[0.97] [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary: "bg-cobalt text-cobalt-ink shadow-sm hover:brightness-110",
        ghost: "text-muted hover:bg-surface-2 hover:text-ink",
        outline: "border border-line bg-surface text-ink hover:bg-surface-2",
        soft: "bg-cobalt-soft text-cobalt hover:brightness-95 dark:hover:brightness-125",
      },
      size: {
        sm: "h-8 px-2.5 text-[13px]",
        md: "h-9 px-3",
        icon: "size-9",
        "icon-sm": "size-8",
      },
    },
    defaultVariants: { variant: "ghost", size: "md" },
  },
);

export type ButtonProps = ComponentProps<"button"> & VariantProps<typeof buttonVariants>;

export function Button({ className, variant, size, type = "button", ...props }: ButtonProps) {
  return <button type={type} className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}
