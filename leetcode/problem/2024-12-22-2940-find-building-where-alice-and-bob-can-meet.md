---
layout: leetcode-entry
title: "2940. Find Building Where Alice and Bob Can Meet"
permalink: "/leetcode/problem/2024-12-22-2940-find-building-where-alice-and-bob-can-meet/"
leetcode_ui: true
entry_slug: "2024-12-22-2940-find-building-where-alice-and-bob-can-meet"
---

[2940. Find Building Where Alice and Bob Can Meet](https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/description/) hard
[blog post](https://leetcode.com/problems/find-building-where-alice-and-bob-can-meet/solutions/6173752/kotlin-rust-by-samoylenkodmitry-8d8q/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/22122024-2940-find-building-where?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/nLRmWDSzYsI)
[deep-dive](https://notebooklm.google.com/notebook/0bce7872-6216-4610-99d9-190532fa9ab2/audio)
![1.webp](/assets/leetcode_daily_images/a9a25d9f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/840

#### Problem TLDR

Common indices t, h[t] > h[a], h[t] > h[b] for queries q[][a,b] #hard #monotonic_stack

#### Intuition

Didn't solve it without a hint.
The hint: consider queries by rightmost border, use monotonic stack, binary search in it.

Let's observe an example:

```j

    // 0 1 2 3 4 5 6 7
    // 5 3 8 2 6 1 4 6
    // a             b*
    //   a         b*
    //       a   b *>2  1 4 6
    //     b-    a>8
    // b     a *>5     2 6

    // a b [8 2 1], [8]
    // i

```

* we can walk height from the end
* for each right border of a query we should find the closest height that is bigger than `a` and `b`
* so we should keep big numbers, pop all smaller

Some meta-thoughts: I have considered the monotonic stack/queue, but the solution requiers another leap of insight, the Binary Search. So, this is a two-level-deep insight problem.

#### Approach

* to have an intuition about what kind of monotonic stack needed, ask `what numbers are useful for the current situation, and what aren't?`

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun leftmostBuildingQueries(heights: IntArray, queries: Array<IntArray>): IntArray {
        val r = IntArray(queries.size); val q = ArrayList<Int>(); var j = heights.lastIndex
        for (i in queries.indices.sortedBy { -queries[it].max() }) {
            val (a, b) = (queries[i].min() to queries[i].max())
            if (a == b || heights[a] < heights[b]) { r[i] = b; continue }
            while (j > b) {
                while (q.size > 0 && heights[q.last()] < heights[j]) q.removeLast()
                q += j--
            }
            var lo = 0; var hi = q.lastIndex; r[i] = -1
            while (lo <= hi) {
                val m = lo + (hi - lo) / 2
                if (heights[q[m]] > heights[a]) { r[i] = q[m]; lo = m + 1 }
                else hi = m - 1
            }
        }
        return r
    }

```
```rust

    pub fn leftmost_building_queries(heights: Vec<i32>, queries: Vec<Vec<i32>>) -> Vec<i32> {
        let (mut r, mut q, mut j) = (vec![-1; queries.len()], vec![], heights.len() - 1);
        let mut qu: Vec<_> = (0..r.len()).map(|i| { let q = &queries[i]; (-q[0].max(q[1]), q[0].min(q[1]), i)}).collect();
        qu.sort_unstable();
        for (b, a, i) in qu {
            let (a, b) = (a as usize, (-b) as usize);
            if a == b || heights[a] < heights[b] { r[i] = b as i32; continue }
            while j > b {
                while q.last().map_or(false, |&l| heights[l] < heights[j]) { q.pop(); }
                q.push(j); j -= 1;
            }
            let (mut lo, mut hi) = (0, q.len() - 1);
            while lo <= hi && hi < q.len() {
                let m = lo + (hi - lo) / 2;
                if heights[q[m]] > heights[a] { r[i] = q[m] as i32; lo = m + 1 }
                else { hi = m - 1 }
            }
        }; r
    }

```
```c++

    vector<int> leftmostBuildingQueries(vector<int>& hs, vector<vector<int>>& qs) {
        vector<int> q, idx, r(qs.size()); int j = hs.size() - 1;
        for (int i = 0; i < qs.size(); ++i) {
            sort(begin(qs[i]), end(qs[i]));
            if (qs[i][0] == qs[i][1] || hs[qs[i][0]] < hs[qs[i][1]]) r[i] = qs[i][1];
            else idx.push_back(i);
        }
        sort(begin(idx), end(idx), [&](int i, int j) { return qs[i][1] > qs[j][1]; });
        for (int i: idx) {
            int a = qs[i][0], b = qs[i][1];
            while (j > b) {
                while (q.size() && hs[q.back()] <= hs[j]) q.pop_back();
                q.push_back(j--);
            }
            auto it = upper_bound(rbegin(q), rend(q), a, [&](int i, int j) { return hs[i] < hs[j]; });
            r[i] = it == rend(q) ? -1 : *it;
        } return r;
    }

```

