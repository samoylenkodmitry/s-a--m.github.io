---
layout: leetcode-entry
title: "3691. Maximum Total Subarray Value II"
permalink: "/leetcode/problem/2026-06-10-3691-maximum-total-subarray-value-ii/"
leetcode_ui: true
entry_slug: "2026-06-10-3691-maximum-total-subarray-value-ii"
---

[3691. Maximum Total Subarray Value II](https://leetcode.com/problems/maximum-total-subarray-value-ii/solutions/8325421/kotlin-rust-by-samoylenkodmitry-nvmh/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10062026-3691-maximum-total-subarray?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/1YRY30jcew4)

https://dmitrysamoylenko.com/leetcode/

![10.06.2026.webp](/assets/leetcode_daily_images/10.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1386

#### Problem TLDR

sum k largest ranges max(l..r)-min(l..r)

#### Intuition

* segment tree: size is 2*n, right lalf is the leafs, left half is the parents; to query look when left pointer is the right child, and right pointer is the left child

#### Approach

1. Use segment tree to query max and in of range l..r
2. The largest max-min is when the range is the widest, put all widest ranges to a sorted collection
3. Query sorted collection and update with shrinked range l..r-1

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maxTotalValue(n: IntArray, k: Int): Long {
        val s = n.size; val t0 = IntArray(s*2); val t1 = IntArray(s*2)
        for (i in n.indices) { t0[i+s] = n[i]; t1[i+s] = n[i] }
        for (i in s-1 downTo 1) { t0[i]=min(t0[i*2], t0[i*2+1]); t1[i]=max(t1[i*2], t1[i*2+1]) }
        fun q(l: Int, r: Int): IntArray {
            var a=l+s; var b=r+s; var m=Int.MAX_VALUE; var M=0
            while (a <= b) {
                if (a % 2 > 0) { m = min(m, t0[a]); M = max(M, t1[a++]) }
                if (b % 2 < 1) { m = min(m, t0[b]); M = max(M, t1[b--]) }
                a /= 2; b /= 2
            }
            return intArrayOf(l, r, M - m)
        }
        val pq = PriorityQueue<IntArray>(compareBy { -it[2] })
        for (i in n.indices) pq += q(i, s-1)
        return (1..k).sumOf { pq.poll()?.let {(l,r,v) -> if (r>l) pq += q(l,r-1); 1L*v } ?: 0L }
    }
```
```rust
    pub fn max_total_value(n: Vec<i32>, k: i32) -> i64 {
        let s = n.len(); let (mut t0, mut t1) = (vec![0; s * 2], vec![0; s * 2]);
        for i in 0..s { t0[i+s] = n[i]; t1[i+s] = n[i] }
        for i in (1..s).rev() {  t0[i]=t0[i*2].min(t0[i*2+1]); t1[i]=t1[i*2].max(t1[i*2+1])}
        let q = |l: usize, r: usize| {
            let (mut a, mut b, mut m, mut M) = (l+s, r+s, i32::MAX, i32::MIN);
            while a <= b {
                if a % 2 > 0 { m = m.min(t0[a]); M = M.max(t1[a]); a += 1 }
                if b % 2 == 0 { m = m.min(t0[b]); M = M.max(t1[b]); b -= 1 }
                a /= 2; b /= 2
            }
            (M - m, l, r)
        };
        let mut pq: BinaryHeap<_> = (0..s).map(|i| q(i, s - 1)).collect();
        (0..k).filter_map(|_|pq.pop().map(|(v,l,r)|{if r>l{pq.push(q(l,r-1))};v as i64})).sum()
    }
```

