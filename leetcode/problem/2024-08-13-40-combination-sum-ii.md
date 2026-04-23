---
layout: leetcode-entry
title: "40. Combination Sum II"
permalink: "/leetcode/problem/2024-08-13-40-combination-sum-ii/"
leetcode_ui: true
entry_slug: "2024-08-13-40-combination-sum-ii"
---

[40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/description/) medium
[blog post](https://leetcode.com/problems/combination-sum-ii/solutions/5629297/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13082024-40-combination-sum-ii?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ikn6xyVEQAQ)
![1.webp](/assets/leetcode_daily_images/67a2e282.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/701

#### Problem TLDR

Unique target sum subsequences #medium #backtracking

#### Intuition

Let's start from the brute force backtracking solution: start with some index and choose which index would be the next.

The interesting part is how to handle duplicates. Simple HashSet gives TLE.

Let's look at the example `1 1 1 2`: each `1` start the same sequence `1 2`, so we can skip the second and the third `1`'s.

#### Approach

* we can use slices in Rust instead of a pointer
* minor optimization is breaking early when the sum is overflown

#### Complexity

- Time complexity:
$$O(n^n)$$

- Space complexity:
$$O(n^n)$$

#### Code

```kotlin

    fun combinationSum2(candidates: IntArray, target: Int): List<List<Int>> = buildList {
        val curr = mutableListOf<Int>(); candidates.sort()
        fun dfs(i: Int, t: Int): Unit = if (t == 0) { add(curr.toList()); Unit }
            else for (j in i..<candidates.size) {
                if (j > i && candidates[j] == candidates[j - 1]) continue
                if (candidates[j] > t) break
                curr += candidates[j]
                dfs(j + 1, t - candidates[j])
                curr.removeLast()
            }
        dfs(0, target)
    }

```
```rust

    pub fn combination_sum2(mut candidates: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
        candidates.sort_unstable();
        fn dfs(c: &[i32], t: i32, res: &mut Vec<Vec<i32>>, curr: &mut Vec<i32>) {
            if t == 0 { res.push(curr.clone()); return }
            for j in 0..c.len() {
                if j > 0 && c[j] == c[j - 1] { continue }
                if c[j] > t { break }
                curr.push(c[j]);
                dfs(&c[j + 1..], t - c[j], res, curr);
                curr.remove(curr.len() - 1);
            }
        }
        let (mut res, mut curr) = (vec![], vec![]);
        dfs(&candidates, target, &mut res, &mut curr); res
    }

```

