---
layout: leetcode-entry
title: "2106. Maximum Fruits Harvested After at Most K Steps"
permalink: "/leetcode/problem/2025-08-03-2106-maximum-fruits-harvested-after-at-most-k-steps/"
leetcode_ui: true
entry_slug: "2025-08-03-2106-maximum-fruits-harvested-after-at-most-k-steps"
---

[2106. Maximum Fruits Harvested After at Most K Steps](https://leetcode.com/problems/maximum-fruits-harvested-after-at-most-k-steps/description/) hard
[blog post](https://leetcode.com/problems/maximum-fruits-harvested-after-at-most-k-steps/solutions/7038844/kotlin-rust-by-samoylenkodmitry-2eg7/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/3082025-2106-maximum-fruits-harvested?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/m5BB3noDmO0)
![1.webp](/assets/leetcode_daily_images/327ecfef.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1069

#### Problem TLDR

Max values k steps from start point #hard #sliding_window

#### Intuition

```j
    // k = 30
    // 0      1     2      3      4      5      6      7      8      9      10     11     12     13      14     15      16    17      18
    // [[0,7],[7,4],[9,10],[12,6],[14,8],[16,5],[17,8],[19,4],[20,1],[21,3],[24,3],[25,3],[26,1],[28,10],[30,9],[31,6],[32,1],[37,5],[40,9]]
    //                                                               sp

```
Sliding window:
* always move the right border
* move left until condition
* update max

The main hardness is to peek the smaller path between the two: go back then forward, or go forward then back.

#### Approach

* the queue is not necessary, just a left pointer is enough

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 15ms
    fun maxTotalFruits(f: Array<IntArray>, sp: Int, k: Int): Int {
        var q = 0; var sum = 0
        return f.maxOf { (x, a) ->
            while (min(sp + x - 2*f[q][0], 2*x - sp - f[q][0]) > k) sum -= f[q++][1]
            if (abs(x - sp) > k) { ++q; 0 } else { sum += a; sum }
        }
    }

```
```kotlin

// 3ms
    fun maxTotalFruits(f: Array<IntArray>, sp: Int, k: Int): Int {
        var q = 0; var sum = 0; var r = 0
        for ((x, a) in f) {
            if (x - sp > k) break; if (sp - x > k) { ++q; continue }
            while (min(sp + x - 2*f[q][0], 2*x - sp - f[q][0]) > k) sum -= f[q++][1]
            sum += a; r = max(r, sum)
        }
        return r
    }

```
```rust

// 11ms
    pub fn max_total_fruits(f: Vec<Vec<i32>>, sp: i32, k: i32) -> i32 {
        let (mut q, mut s) = (0, 0);
        f.iter().map(|v| {
            while (sp + v[0] - 2*f[q][0]).min(2*v[0] - sp - f[q][0]) > k { s -= f[q][1]; q += 1 }
            if (v[0] - sp).abs() > k { q += 1; 0 } else { s += v[1]; s }
        }).max().unwrap()
    }

```
```c++

// 2ms
    int maxTotalFruits(vector<vector<int>>& f, int sp, int k) {
        int q = 0, s = 0, r = 0;
        for (auto& v: f) {
            if (v[0] - sp > k) break; if (sp - v[0] > k) { ++q; continue; }
            while (min(sp + v[0] - 2*f[q][0], 2*v[0] - sp - f[q][0]) > k) s -= f[q++][1];
            r = max(r, s += v[1]);
        } return r;
    }

```
```python

// 104ms
    def maxTotalFruits(self, f: List[List[int]], s: int, k: int) -> int:
        q = r = t = 0
        for x, a in f:
            if s - x > k: q += 1; continue
            while min(s + x - 2*f[q][0], 2*x - s - f[q][0]) > k: t -= f[q][1]; q += 1
            t += a if x - s <= k else 0; r = max(r, t)
        return r

```

