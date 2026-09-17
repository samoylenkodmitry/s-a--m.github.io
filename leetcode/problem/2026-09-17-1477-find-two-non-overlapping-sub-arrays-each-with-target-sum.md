---
layout: leetcode-entry
title: "1477. Find Two Non-overlapping Sub-arrays Each With Target Sum"
permalink: "/leetcode/problem/2026-09-17-1477-find-two-non-overlapping-sub-arrays-each-with-target-sum/"
leetcode_ui: true
entry_slug: "2026-09-17-1477-find-two-non-overlapping-sub-arrays-each-with-target-sum"
---

[1477. Find Two Non-overlapping Sub-arrays Each With Target Sum](https://leetcode.com/problems/find-two-non-overlapping-sub-arrays-each-with-target-sum/solutions/8526126/kotlin-rust-by-samoylenkodmitry-7fzn/) medium
[substack](https://dmitriisamoilenko.substack.com/p/17092026-1477-find-two-non-overlapping?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/l-eefPKWwEc)

https://dmitrysamoylenko.com/leetcode/

![17.09.2026.webp](/assets/leetcode_daily_images/17.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1485

#### Problem TLDR

Two min length subarrays equal target

#### Intuition

Build for prefix the so-far minimum length of subarray. Same for suffix. Then check each position by prefix+suffix.
Sliding window: store the minimum length subarray prefix, shrink window to be equal to target, then current window would not intersect the prefix[i]+(r-l+1)

#### Approach

* reuse building function of the prefix for the suffix

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun minSumOfLengths(a: IntArray, t: Int) = run {
        val dp = IntArray(a.size+1){a.size+1}; var s = 0; var l = 0
        a.indices.minOf { r ->
            s += a[r]; while (s > t) s -= a[l++]
            dp[r + 1] = if (s < t) dp[r] else min(dp[r], r - l + 1)
            if (s == t) dp[l] + r - l + 1 else a.size+1
        }.takeIf { it <= a.size } ?: -1
    }
```
```rust
    pub fn min_sum_of_lengths(a: Vec<i32>, t: i32) -> i32 {
        let (n, mut s, mut l, mut m)=(a.len(),0,0,a.len()+1); let mut d=vec![m; m];
        for r in 0..n {
            s += a[r]; while s > t { s -= a[l]; l += 1 }
            d[r + 1] = if s == t { m = m.min(r+1-l + d[l]); d[r].min(r+1-l) } else { d[r] }
        }
        if m <= n { m as _ } else { -1 }
    }
```

