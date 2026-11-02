import { useQuery } from "@tanstack/react-query";
import { apiGet, type CategoriesResp } from "./api";

export function useCategories() {
  return useQuery({ queryKey: ["nlp-categories"], queryFn: () => apiGet<CategoriesResp>("/nlp/categories"), staleTime: Infinity });
}

export function categoryLabel(id: string | null | undefined, cats?: CategoriesResp): string {
  if (!id) return "–";
  if (id === "abstain") return "Abstained";
  return cats?.categories.find((c) => c.id === id)?.label ?? id;
}
