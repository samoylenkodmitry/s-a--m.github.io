---
layout: leetcode-entry
title: "2163. Minimum Difference in Sums After Removal of Elements"
permalink: "/leetcode/problem/2025-07-18-2163-minimum-difference-in-sums-after-removal-of-elements/"
leetcode_ui: true
entry_slug: "2025-07-18-2163-minimum-difference-in-sums-after-removal-of-elements"
---

[2163. Minimum Difference in Sums After Removal of Elements](https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/description/) hard
[blog post](https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/solutions/6973297/kotlin-rust-by-samoylenkodmitry-b8oo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18072025-2163-minimum-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/XqGjzlfXcXU)
![1.webp](/assets/leetcode_daily_images/f40caa78.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1053

#### Problem TLDR

Min first half - second half after removing n/3 #hard #heap

#### Intuition

Used the hint:
* consider each index as partition point, `min sum before - max sum after`

```j

    // 795813
    // remove big left, small right
    // order inside parts doesn't matter
    // left = 795     right = 813
    // but if removed parts diff > 2 number migrates
    //          5             813
    //                 8 goes left
    //          58            13

    // 991199
    // 991   199    d =9+9+1 - 1 +9 +9 = 19 - 19 = 0
    // 9 91  199    d = 9+1 - 1+9+9 = 10-19 = -9
    // 99 1  199    d = 1 - 1+9+9 = -18, but 1 goes left
    //                  1+1 - 9+9 = -16
    //
    //   11 99
    //            ok but how to shift mid if it was removed? (21 minute)
    // used hints: find min/max n-sum for prefix/suffix for every i

```

My own idea was to maintain two heaps and poll elements from them, but I was stuck with how to balance them correctly.

#### Approach

* carefull with off-by-ones, sums should not overlap

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$, or O(n / 3)

#### Code

```kotlin

// 109ms
    fun minimumDifference(n: IntArray): Long {
        val q = PriorityQueue<Int>(); val suf = LongArray(n.size)
        var s = 0L; val k = n.size / 3
        for (i in n.lastIndex downTo k) {
            q += n[i]; s += n[i]; if (q.size > k) s -= q.poll()
            suf[i] = s
        }
        q.clear(); var d = s; s = 0
        for (i in 0..<2 * k) {
            q += -n[i]; s += n[i]; if (q.size > k) s += q.poll()
            if (i >= k - 1) d = min(d, s - suf[i + 1])
        }
        return d
    }

```

```rust

// 47ms
    pub fn minimum_difference(n: Vec<i32>) -> i64 {
        let (mut q, k) = (BinaryHeap::new(), n.len() / 3);
        let (mut suf, mut s, mut j) = (vec![0; k + 1], 0i64, 0);
        for i in (k..n.len()).rev() { let n = n[i] as i64;
            q.push(-n); s += n; if q.len() > k { s += q.pop().unwrap(); }
            if q.len() == k { suf[j] = s; j += 1 }
        }
        q.clear(); let mut d = s; s = 0;
        for i in 0..2 * k { let n = n[i] as i64;
            q.push(n); s += n; if q.len() > k { s -= q.pop().unwrap(); }
            if i >= k - 1 { j -= 1; d = d.min(s - suf[j]) }
        } d
    }

```
```c++

// 131ms
    long long minimumDifference(const vector<int>& n) {
        int k = n.size() / 3, j = 0; long long s = 0;
        priority_queue<int> q; vector<long long> suf(k + 1);
        for (int i = n.size() - 1; i >= k; --i) {
            s += n[i]; q.push(-n[i]);
            if (q.size() > k) s += q.top(), q.pop();
            if (q.size() == k) suf[j++] = s;
        }
        q = {}; long long d = s; s = 0;
        for (int i = 0; i < 2 * k; ++i) {
            s += n[i]; q.push(n[i]);
            if (q.size() > k) s -= q.top(), q.pop();
            if (i >= k - 1) d = min(d, s - suf[--j]);
        }
        return d;
    }

```

