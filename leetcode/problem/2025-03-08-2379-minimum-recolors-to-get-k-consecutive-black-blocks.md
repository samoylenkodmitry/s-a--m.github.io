---
layout: leetcode-entry
title: "2379. Minimum Recolors to Get K Consecutive Black Blocks"
permalink: "/leetcode/problem/2025-03-08-2379-minimum-recolors-to-get-k-consecutive-black-blocks/"
leetcode_ui: true
entry_slug: "2025-03-08-2379-minimum-recolors-to-get-k-consecutive-black-blocks"
---

[2379. Minimum Recolors to Get K Consecutive Black Blocks](https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks/description) easy
[blog post](https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks/solutions/6512098/kotlin-rust-by-samoylenkodmitry-1ntp/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08032025-2379-minimum-recolors-to?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/JQVEn1qfrs4)
![1.webp](/assets/leetcode_daily_images/3c02af60.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/919

#### Problem TLDR

Min 'W' flips to make k 'B' #medium #sliding_window

#### Intuition

The brute-force is accepted: count 'W' in k-length slice from every index.

Another solution is a sliding window: move k-length window and count 'W'.

#### Approach

* the best way to not make one-off mistake is to avoid pointes at all

#### Complexity

- Time complexity:
$$O(n^2)$$, or O(n)

- Space complexity:
$$O(n)$$, or O(1)

#### Code

```kotlin

    fun minimumRecolors(blocks: String, k: Int) =
        blocks.windowed(k).minOf { it.count { it > 'B' }}

```
```rust

    pub fn minimum_recolors(blocks: String, k: i32) -> i32 {
        let (s, mut w, k) =  (blocks.as_bytes(), 0, k as usize);
        (0..s.len()).map(|r| {
            if s[r] > b'B' { w += 1 }
            if r + 1 < k { 100 } else
            if s[r + 1 - k] > b'B' { w -= 1; w + 1 } else { w }
        }).min().unwrap()
    }

```
```c++

    int minimumRecolors(string b, int k, int res = 100, int w = 0) {
        for (int r = 0; r < size(b); w -= ++r >= k && b[r - k] > 'B')
            w += b[r] > 'B', res = min(res, r < k - 1 ? 100 : w);
        return res;
    }

```

