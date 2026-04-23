---
layout: leetcode-entry
title: "3296. Minimum Number of Seconds to Make Mountain Height Zero"
permalink: "/leetcode/problem/2026-03-13-3296-minimum-number-of-seconds-to-make-mountain-height-zero/"
leetcode_ui: true
entry_slug: "2026-03-13-3296-minimum-number-of-seconds-to-make-mountain-height-zero"
---

[3296. Minimum Number of Seconds to Make Mountain Height Zero](https://open.substack.com/pub/dmitriisamoilenko/p/13032026-3296-minimum-number-of-seconds?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) medium
[blog post](https://open.substack.com/pub/dmitriisamoilenko/p/13032026-3296-minimum-number-of-seconds?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13032026-3296-minimum-number-of-seconds?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ulyJXpETKLc)

![ff059a99-0832-43e9-a470-b09da8c22eb4 (1).webp](/assets/leetcode_daily_images/1c267ce3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1296

#### Problem TLDR

Min steps to work a mountain with arithmetic-sum-workers #medium #bs

#### Intuition

```
    // the description is brainrot
    // 1+2+3+4+5 + (n-2)+(n-1)+n = s
    //
```
Binary search: freeze the number of parallel steps, test the amount of total work done.
Inner binary search to answer what number of steps is the maximum allowed to be less than frozen steps.

#### Approach

* S = x*(x+1)/2
* there is a math solution possible for inner bs
* live math in rust is slower than precomputed array in kotlin

#### Complexity

- Time complexity:
$$O(log(nlog(w)))$$

- Space complexity:
$$O(w)$$ can be O(1)

#### Code

```kotlin
// 117ms
    fun minNumberOfSeconds(mh: Int, wt: IntArray): Long {
        val a = LongArray(mh+1) { 1L * it * (it+1)/2 }
        var lo = 0L; var hi = 1L shl 62
        while (lo <= hi) {
            val m = lo + (hi - lo) / 2
            if (mh > wt.sumOf { t ->
                var i = a.binarySearch(m/t); if (i < 0) i = -i-2; i
            }) lo = m + 1 else hi = m - 1
        }
        return lo
    }
```
```rust
// 144ms
    pub fn min_number_of_seconds(mh: i32, wt: Vec<i32>) -> i64 {
        let (mut lo, mut hi) = (0i64, 1i64<<62);
        while lo <= hi {
            let m = lo + (hi - lo) / 2; let mut s = 0i64;
            for &t in &wt {
                let (mut l, mut h) = (0, mh as i64);
                while l <= h {
                    let i = (l + h) / 2;
                    if i*(i+1)/2 * t as i64 <= m { l = i + 1 } else { h = i - 1 }
                }
                s += h
            }
            if s < mh as i64 { lo = m + 1 } else { hi = m - 1 }
        } lo
    }
```

