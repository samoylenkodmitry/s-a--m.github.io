---
layout: leetcode-entry
title: "960. Delete Columns to Make Sorted III"
permalink: "/leetcode/problem/2025-12-22-960-delete-columns-to-make-sorted-iii/"
leetcode_ui: true
entry_slug: "2025-12-22-960-delete-columns-to-make-sorted-iii"
---

[960. Delete Columns to Make Sorted III](https://leetcode.com/problems/delete-columns-to-make-sorted-iii/description/) hard
[blog post](https://leetcode.com/problems/delete-columns-to-make-sorted-iii/solutions/7430718/kotlin-rust-by-samoylenkodmitry-09v1/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22122025-960-delete-columns-to-make?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/6wFRYLWGGgU)

![d2dae9ed-5e4f-4520-87e5-b73d466bacf1 (1).webp](/assets/leetcode_daily_images/719db376.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1212

#### Problem TLDR

Remove columns to sort strings #hard #dp

#### Intuition

```j
    // a b c
    // c b c
    // c a b
    //
    // i think just removing column greedily is not optimal
    //
    // a b c d e a
    // e a b c d e

    // **..*** we have sorted and usorted parts
    // *.**... they can overlap
    // abxycde find minimum to remove to make sorted? LIS?
    //         acceptance rate is 67% am i overthinking it?
    //
    // maybe it is dp, the tail should be all sorted
    // 28minute; look at hints; ok it is a LIS and is a DP
```

Top down dp: take i or skip it, compare with previous j.

#### Approach

* 1-D dp, build longest substring: lookup prefixes to increase the length dp[j]+1
* 1-D dp, minimum removals: lookup prefixes to find min removals dp[j]+i-j-1 (more tricky with initial conditions)

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n^2)$$ or n^2 optimized

#### Code

```kotlin
// 20ms
    fun minDeletionSize(s: Array<String>): Int {
        val d = HashMap<Int, Int>()
        fun dfs(i: Int, j: Int): Int = if (i == s[0].length) 0 else d.getOrPut(i*100+j) {
            val skip = 1 + dfs(i + 1, j)
            val take = if (j < 0 || s.all {it[j] <= it[i]}) dfs(i+1,i) else 100
            min(skip, take)
        }
        return dfs(0, -1)
    }
```
```rust
// 1ms
    pub fn min_deletion_size(s: Vec<String>) -> i32 {
        let m = s[0].len(); let mut d: Vec<_> = (0..m+2).collect();
        for i in 2..m+2 { for j in 1..i {
            if i > m || s.iter().all(|r| r[j-1..=j-1] <= r[i-1..=i-1]) {
                d[i] = d[i].min(d[j] + i - j - 1);
        }}} d[m+1] as i32 -1
    }
```

