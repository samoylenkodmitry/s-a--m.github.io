---
layout: leetcode-entry
title: "3721. Longest Balanced Subarray II"
permalink: "/leetcode/problem/2026-02-11-3721-longest-balanced-subarray-ii/"
leetcode_ui: true
entry_slug: "2026-02-11-3721-longest-balanced-subarray-ii"
---

[3721. Longest Balanced Subarray II](https://leetcode.com/problems/longest-balanced-subarray-ii/description) hard
[blog post](https://leetcode.com/problems/longest-balanced-subarray-ii/solutions/7570897/kotlin-rust-by-samoylenkodmitry-rx7o/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11022026-3721-longest-balanced-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/LuRCi31Yl_U)

![a0f6bd95-d9f3-467a-b781-d170cc970ba3 (1).webp](/assets/leetcode_daily_images/d6dc4ac8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1265

#### Problem TLDR

Max subarray evens count = odds count #hard #segment_tree

#### Intuition

Didn't solve.
```j
    // this is a hard version of yesterdays problem
    // O(n^2) will not be accepted
    //
    // my idea: define window size and binary search it
    // when moving the window we have to adjust count on the fly
    //
    //    3 2 2 5 4    w=2
    //    * *          f[3]=1 f[2]=1
    //      * *        f[3]=0 f[2]=2   (and count uniqs in parallel)
    //
    // stop: does this function linear from true to false?
    //     small subarray is NOT balanced
    //     and big subarray IS balanced
    //                so the binary search wouldn't work
    //
    // given the small acceptance rate i go for hints at 10 minute
    // so its a segment tree;
```
0. Convert evens and odds to +1/-1
1. Segment Tree: manage the prefix sum up to i
2. If sum[root] == 0, then entire prefix is balanced
3. If sum[root] != 0, find leftmost index `j` with the same value of sum[root], because prefix[i]-prfix[j] = 0 means subarray is balanced
4. Handle duplicated by updating the value to 0 in a segment tree
5. The min[..] and right[..] are the prefix values of subtrees for the search. min[n] is the lowest possible prefix sum in this subtree

#### Approach

* try to understand how this works
* can you answer the question: why we are checking `s in min[left]..right[left]` to go to the left subtree? Why we are shifting min[right]+sum[left]?

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 165ms
    fun longestBalanced(n: IntArray): Int {
        val sz=n.size; val sum = IntArray(sz*4); val min = IntArray(sz*4); val max = IntArray(sz*4)
        fun update(idx: Int, v: Int, l: Int = 0, r: Int = sz-1, n: Int = 1) {
            if (l == r) { sum[n] = v; min[n] = v; max[n] = v; return }
            if (idx <= (l+r)/2) update(idx,v,l,(l+r)/2,n*2) else update(idx,v,(l+r)/2+1,r,n*2+1)
            sum[n] = sum[n*2]+sum[n*2+1]
            min[n] = min(min[n*2], sum[n*2] + min[n*2+1])
            max[n] = max(max[n*2], sum[n*2] + max[n*2+1])
        }
        fun q(l: Int = 0, r: Int = sz-1, n: Int = 1, s: Int = 0): Int = if (l==r) l else
            if (sum[1]-s in min[n*2]..max[n*2]) q(l,(l+r)/2,n*2,s) else q((l+r)/2+1,r,n*2+1,s+sum[n*2])
        val p = IntArray(100001)
        return n.indices.maxOf { i ->
            if (p[n[i]] > 0) update(p[n[i]]-1, 0); p[n[i]] = i+1; update(i, n[i]%2*2-1)
            if (sum[1] == 0) i + 1 else i - q()
        }
    }
```

