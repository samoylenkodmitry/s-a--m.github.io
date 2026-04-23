---
layout: leetcode-entry
title: "3347. Maximum Frequency of an Element After Performing Operations II"
permalink: "/leetcode/problem/2025-10-22-3347-maximum-frequency-of-an-element-after-performing-operations-ii/"
leetcode_ui: true
entry_slug: "2025-10-22-3347-maximum-frequency-of-an-element-after-performing-operations-ii"
---

[3347. Maximum Frequency of an Element After Performing Operations II](https://leetcode.com/problems/maximum-frequency-of-an-element-after-performing-operations-ii/description) hard
[blog post](https://leetcode.com/problems/maximum-frequency-of-an-element-after-performing-operations-ii/solutions/7292282/kotlin-rust-by-samoylenkodmitry-2qsa/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22102025-3347-maximum-frequency-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/O9r2hrtvE5Y)

![3caeb30d-45b6-49e7-8473-6c3f0bc032ab (1).webp](/assets/leetcode_daily_images/b49f1fec.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1150

#### Problem TLDR

Max frequency after changing +k..-k ops elements #medium #sliding_window

#### Intuition

```j
    // its like yesterday problem
    // so we have two numbers 5,11     the k=5 but ops=1
    //                                 we can't change both
    // can we have same situation for 3 numbers?
    //           5,5,11   k=5 ops=2     the strategy min(o,2)*k didn't work
```

Solve two separate problems:
1. choose every number as baseline, window b-k..b+k
2. no baseline, just 2k window

#### Approach

* the first is the frequency of baseline plus all opeations restricted by the window size
* the second is the entire window restricted by the number of operations

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 180ms
    fun maxFrequency(n: IntArray, k: Int, o: Int): Int {
        n.sort(); var l = 0; var j = 0; var r = 0; var f = HashMap<Int,Int>()
        return n.withIndex().maxOf { (i,x) ->
            while (r < n.size && n[r] <= x+k) f[n[r]] = 1 + (f[n[r++]] ?: 0)
            while (l < n.size && n[l] < x-k) f[n[l]] = -1 + f[n[l++]]!!
            while (x - n[j] > 2*k) ++j
            max(min(f[x]!!+o, r-l), min(i-j+1, o))
        }
    }

```
```rust

// 42ms
    pub fn max_frequency(mut n: Vec<i32>, k: i32, o: i32) -> i32 {
        n.sort_unstable(); let (mut l, mut j, mut r, mut f) = (0,0,0,HashMap::new());
        n.iter().enumerate().map(|(i,&x)|{
            while r < n.len() && n[r] <= x+k { *f.entry(n[r]).or_insert(0) += 1; r += 1 }
            while l < n.len() && n[l] < x-k { *f.get_mut(&n[l]).unwrap() -= 1; l += 1 }
            while x - n[j] > 2*k { j += 1 }
            o.min((i-j+1)as i32).max((r-l) as i32).min(f[&x]+o)
        }).max().unwrap_or(0)
    }

```

