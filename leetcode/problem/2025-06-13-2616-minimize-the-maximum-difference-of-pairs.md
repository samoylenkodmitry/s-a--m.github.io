---
layout: leetcode-entry
title: "2616. Minimize the Maximum Difference of Pairs"
permalink: "/leetcode/problem/2025-06-13-2616-minimize-the-maximum-difference-of-pairs/"
leetcode_ui: true
entry_slug: "2025-06-13-2616-minimize-the-maximum-difference-of-pairs"
---

[2616. Minimize the Maximum Difference of Pairs](https://leetcode.com/problems/minimize-the-maximum-difference-of-pairs/description/) medium
[blog post](https://leetcode.com/problems/minimize-the-maximum-difference-of-pairs/solutions/6838635/kotlin-rust-by-samoylenkodmitry-emqy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13062025-2616-minimize-the-maximum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/aaeAn4mgl3M)
![1.webp](/assets/leetcode_daily_images/055890fb.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1018

#### Problem TLDR

Min max diffs non-overlapped pairs #medium #binary_search

#### Intuition

Didn't solve.

```j
    // 1 1 2 3 7 10 10
    // a a b b
    // * x * x   b  b
    //   a a
    //   * x
    // take `p` min diffs
    // 1 1 1 2
    // * .
    //   *
    //     *
    // how to deal with overlaps? maybe gredily skip
    // 18 minute 1, 2, 2, 2, 3, 3, 4 wrong result
    //              a  a     b  b
    // hint1 use DP
    // 36 minute: all hints+TLE
    // 43 minute MLE the hint was misleading
    // hint: binarysearch    (again overlapping pairs?)
    // 54 minute, giveup (ok, missing idea was to solve overlaps by greedily take)

```

The working hint:
* binary search of the max allowed diff
* take diffs greedily from a sorted order

#### Approach

* we can skip `abs`
* exit early on `cnt >= p`
* there is an actual DP solution without MLE

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 31ms
    fun minimizeMax(n: IntArray, p: Int): Int {
        n.sort(); var lo = 0; var hi = n[n.size - 1] - n[0]
        while (lo <= hi) {
            val m = (lo + hi) / 2; var cnt = if (n[n.size - 1] - n[0] <= m) 1 else 0; var i = 1
            while (i < n.size)
                if (n[i] - n[i - 1] > m) i++ else if (++cnt >= p) break else i += 2
            if (cnt >= p) hi = m - 1 else lo = m + 1
        }
        return lo
    }

```
```rust

// 7ms
    pub fn minimize_max(mut n: Vec<i32>, p: i32) -> i32 {
        n.sort_unstable(); let (mut lo, mut hi, l) = (0, n[n.len() - 1] - n[0], n.len());
        while lo <= hi {
            let m = (lo + hi) / 2; let (mut cnt, mut i) = ((m >= n[l - 1] - n[0]) as i32, 1);
            while i < l {
                if m >= n[i] - n[i - 1]
                    { cnt += 1; i += 2; if cnt >= p { break }} else { i += 1 }}
            if cnt >= p { hi = m - 1 } else { lo = m + 1 }
        } lo
    }

```
```c++

// 22ms
    int minimizeMax(vector<int>& n, int p) {
        sort(begin(n), end(n));
        int l = size(n), lo = 0, ld = n.back() - n[0]; int hi = ld;
        while (lo <= hi) {
            int m = (lo + hi) / 2; int c = ld <= m, i = 1;
            while (i < l) if (n[i] - n[i - 1] > m) i++;
                else if (++c >= p) break; else i += 2;
            if (c >= p) hi = m - 1; else lo = m + 1;
        } return lo;
    }

```

