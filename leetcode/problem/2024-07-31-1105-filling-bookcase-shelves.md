---
layout: leetcode-entry
title: "1105. Filling Bookcase Shelves"
permalink: "/leetcode/problem/2024-07-31-1105-filling-bookcase-shelves/"
leetcode_ui: true
entry_slug: "2024-07-31-1105-filling-bookcase-shelves"
---

[1105. Filling Bookcase Shelves](https://leetcode.com/problems/filling-bookcase-shelves/description/) medium
[blog post](https://leetcode.com/problems/filling-bookcase-shelves/solutions/5561897/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/31072024-1105-filling-bookcase-shelves?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/09bHi01l9GY)
![2024-07-31_08-48_1.webp](/assets/leetcode_daily_images/d9b28e67.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/687

#### Problem TLDR

Min total height to split `[[w,h]]` array by `shelfWidth` #medium #dynamic_programming

#### Intuition

Let's do a Depth-First search by the current book position: start forming a shelf by adding books while they are fit into shelfWidth and after each book try to stop and go to the next level dfs. Result is only depends on the starting position, so can be cached.

The bottom up Dynamic Programming algorithm can be thought like this: walk over the books and consider each `i` the end of the array; now choose optimal split `before` in [..i] books but not wider than shelf_width. Previous dp[j] are known, so we can compute `dp[i] = min[h_max + dp[j]]`.

#### Approach

Let's write DFS in Kotlin and bottom-up DP in Rust. Can you make it shorter?

#### Complexity

- Time complexity:
$$O(nm)$$, where m is an average books count on the shelf; O(n^2) for solution without the `break`

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun minHeightShelves(books: Array<IntArray>, shelfWidth: Int): Int {
        val dp = mutableMapOf<Int, Int>()
        fun dfs(j: Int): Int = if (j < books.size) dp.getOrPut(j) {
            var w = 0; var h = 0
            (j..<books.size).minOf { i ->
                w += books[i][0]; h = max(h, books[i][1])
                if (w > shelfWidth) Int.MAX_VALUE else h + dfs(i + 1)
            }
        } else 0
        return dfs(0)
    }

```
```rust

    pub fn min_height_shelves(books: Vec<Vec<i32>>, shelf_width: i32) -> i32 {
        let mut dp = vec![i32::MAX / 2; books.len()];
        for i in 0..dp.len() {
            let mut w = 0; let mut h = 0;
            for j in (0..=i).rev() {
                w += books[j][0];
                if w > shelf_width { break }
                h = h.max(books[j][1]);
                dp[i] = dp[i].min(h + if j > 0 { dp[j - 1] } else { 0 })
            }
        }; dp[dp.len() - 1]
    }

```

