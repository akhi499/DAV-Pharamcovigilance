import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "@/lib/utils";

const alertVariants = cva("relative w-full rounded-lg border px-4 py-3 text-sm [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-3.5 [&>svg~*]:pl-7", {
  variants: {
    variant: {
      default: "bg-card",
      info: "border-primary/30 bg-accent text-accent-foreground",
      warning: "border-signal/40 bg-signal/10",
      destructive: "border-destructive/40 bg-destructive/5 text-destructive",
      provisional: "provisional-frame text-foreground",
    },
  },
  defaultVariants: { variant: "default" },
});

export const Alert = ({ className, variant, ...p }: React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>) => (
  <div role="note" className={cn(alertVariants({ variant }), className)} {...p} />
);
export const AlertTitle = ({ className, ...p }: React.HTMLAttributes<HTMLHeadingElement>) => (
  <h5 className={cn("mb-1 font-semibold leading-none", className)} {...p} />
);
export const AlertDescription = ({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("text-sm leading-relaxed [&_p]:leading-relaxed", className)} {...p} />
);
