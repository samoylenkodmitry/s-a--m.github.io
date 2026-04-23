---
layout: leetcode-entry
title: "2115. Find All Possible Recipes from Given Supplies"
permalink: "/leetcode/problem/2025-03-21-2115-find-all-possible-recipes-from-given-supplies/"
leetcode_ui: true
entry_slug: "2025-03-21-2115-find-all-possible-recipes-from-given-supplies"
---

[2115. Find All Possible Recipes from Given Supplies](https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies/description/) medium
[blog post](https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies/solutions/6562294/kotlin-rust-by-samoylenkodmitry-gu5g/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21032025-2115-find-all-possible-recipes?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6yeIg3JOVF8)
![1.webp](/assets/leetcode_daily_images/04127d66.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/934

#### Problem TLDR

Valid recipes from supplies #medium #dfs #dp

#### Intuition

Filter out:
* foreign words, not in the recipes or supplies
* cycles

#### Approach

* we can memoize the already checked recipes

#### Complexity

- Time complexity:
$$O(n)$$, all the recipes are visited at most once

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun findAllRecipes(rec: Array<String>, ing: List<List<String>>, sup: Array<String>): List<String> {
        val dp = HashMap<String, Boolean>(); val sup = sup.toSet(); val rs = rec.indices.associate { rec[it] to it }
        fun dfs(r: String): Boolean = r in sup || r in rs && dp.getOrPut(r) { dp[r] = false; ing[rs[r]!!].all(::dfs) }
        return rec.filter(::dfs)
    }

```
```rust

    pub fn find_all_recipes(rec: Vec<String>, ing: Vec<Vec<String>>, sup: Vec<String>) -> Vec<String> {
        let mut dp = vec![0; rec.len()];
        fn dfs(r: &String, dp: &mut Vec<usize>, rec: &Vec<String>, ing: &Vec<Vec<String>>, sup: &Vec<String>) -> bool {
            sup.contains(r) || { if let Some(i) = (0..rec.len()).find(|&x| rec[x] == *r) {
                dp[i] != 1 && (dp[i] > 1 || { dp[i] = 1; if ing[i].iter().all(|x| dfs(x, dp, rec, ing, sup)) { dp[i] = 2 } dp[i] > 1 })
            } else { false }}}
        rec.iter().filter(|r| dfs(r, &mut dp, &rec, &ing, &sup)).collect::<Vec<_>>().into_iter().cloned().collect()
    }

```
```c++

    vector<string> findAllRecipes(vector<string>& rec, vector<vector<string>>& ing, vector<string>& sup) {
        unordered_set<string> sups(begin(sup), end(sup)); unordered_map<string, int>rs, dp; vector<string> a;
        for (int i = 0; i < size(rec); ++i) rs[rec[i]] = i;
        auto dfs = [&](this const auto& dfs, string& r) -> int {
            if (sups.count(r)) return 1; if (!rs.count(r)) return 0; if (dp.count(r)) return dp[r];
            dp[r] = 0; for (auto& i: ing[rs[r]]) if (!dfs(i)) return 0;
            return dp[r] = 1;
        };
        for (auto& r: rec) if (dfs(r)) a.push_back(r); return a;
    }

```

