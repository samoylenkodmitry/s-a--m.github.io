---
layout: leetcode-entry
title: "2501. Longest Square Streak in an Array"
permalink: "/leetcode/problem/2024-10-28-2501-longest-square-streak-in-an-array/"
leetcode_ui: true
entry_slug: "2024-10-28-2501-longest-square-streak-in-an-array"
---

[2501. Longest Square Streak in an Array](https://leetcode.com/problems/longest-square-streak-in-an-array/description/) medium
[blog post](https://leetcode.com/problems/longest-square-streak-in-an-array/solutions/5977610/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28102024-2501-longest-square-streak?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DwMk5yxD_hs)
[deep-dive](https://notebooklm.google.com/notebook/d4e4dbca-8d04-400d-ba26-25de91bc2a86/audio)
![1.webp](/assets/leetcode_daily_images/f3ecfa04.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/783

#### Problem TLDR

Longest quadratic subset #medium #hashmap #math

#### Intuition

Let's look at the problem:

```j

    [4,3,6,16,8,2]
     *               2 or 8
       *             9
         *           36
           *         4 or 256
              *      64
                *    4

```
For each number `n` we want to know if any `n^2` or `sqrt(n)` is present. We can use a HashMap to store that fact.
Other interesting notes:
* in increasing order, we only care about one next number `n^2`
* the problem set is `10^5`, the biggest `n^2 = 316 * 316`, we can search just `2..316` range

#### Approach

* let's do a sorting + hashmap solution in Kotlin, and optimized solution in Rust
* careful with an int overflow

#### Complexity

- Time complexity:
$$O(nlog(n))$$ or O(n)

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun longestSquareStreak(nums: IntArray): Int {
        val streak = mutableMapOf<Int, Int>()
        return nums.sorted().maxOf { n ->
            (1 + (streak[n] ?: 0)).also { streak[n * n] = it }
        }.takeIf { it > 1 } ?: -1
    }

```
```rust

    pub fn longest_square_streak(nums: Vec<i32>) -> i32 {
        let (mut set, mut vmax, mut max) = ([0; 316 * 316 + 1], 0, -1);
        for n in nums { let n = n as usize; if n < set.len() {
            set[n] = 1; vmax = vmax.max(n);
        }}
        for start in 2..317 { if set[start] > 0 {
            let (mut sq, mut streak) = (start * start, 1);
            while 0 < sq && sq <= vmax && set[sq] > 0 {
                streak += 1; sq = sq * sq; max = max.max(streak)
            }
        }}; max
    }

```
```c++

    int longestSquareStreak(vector<int>& nums) {
        int set[316 * 316 + 1] = {}, vmax = 0, res = -1;
        for (int n: nums) if (n <= 316 * 316) set[n] = 1, vmax = max(vmax, n);
        for (int start = 2; start < 317; ++start) if (set[start]) {
            long sq = start * start; int streak = 1;
            while (sq <= vmax && set[sq]) ++streak, sq *= sq, res = max(res, streak);
        }
        return res;
    }

```

