---
layout: leetcode-entry
title: "3161. Block Placement Queries"
permalink: "/leetcode/problem/2026-05-30-3161-block-placement-queries/"
leetcode_ui: true
entry_slug: "2026-05-30-3161-block-placement-queries"
---

[3161. Block Placement Queries](https://leetcode.com/problems/block-placement-queries/solutions/8302367/kotlin-rust-by-samoylenkodmitry-edzo/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/30052026-3161-block-placement-queries?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/RxfxXaQ3IRM)

https://dmitrysamoylenko.com/leetcode/

![30.05.2026.webp](/assets/leetcode_daily_images/30.05.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1375

#### Problem TLDR

Queries 'can place a gap after place an obstacle'

#### Intuition

Didn't solved without a hint.
The TreeMap solution:
* group starting points per gap size
* store obstacles in sorted set
* query: fine left and right obstacle, query the gap, take first starting poing, check size
* build: find left and right obstacle, remove old starting point from that gap, add new starting points to left and right gaps

The SegmentTree solution:
* role: finds the max gap that starts in range 0..x
* build: put current T[M+x] = sz, propagate to parents
* save obstacles in a sorted set
* query: find left obstacle, one gap is x-l, second gap is segment tree maximum in range 0..l-1 (corner case excludes l)

#### Approach

* another segment tree solution is defined by role: find the max gap in range 0..x (it requires to store [start,end,max] per node)

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun getResults(q: Array<IntArray>) = sortedSetOf(0, 100000).run {
        val gs = TreeMap(mapOf(-100000 to sortedSetOf(0)))
        q.mapNotNull { q -> val x = q[1]
            if (q[0] > 1) gs.any { (g, s) -> -g >= q[2] && s.first() <= x-q[2] }
            else null.also { val l = lower(x); val r = higher(x); add(x)
                gs[l-r]?.let { it -= l; if (it.isEmpty()) gs -= l-r }
                gs.getOrPut(l-x, ::TreeSet) += l; gs.getOrPut(x-r, ::TreeSet) += q[1]
            }
        }
    }
```
```rust
    pub fn get_results(q: Vec<Vec<i32>>) -> Vec<bool> {
        let (mut o, mut t, z) = (BTreeSet::from([0, 50005]), vec![0; 100010], 50005); t[z]=z as i32;
        q.into_iter().filter_map(|c| { let x=c[1];
            if c[0]>1 { let f=*o.range(..=x).next_back()?; let (mut l, mut r, mut m)=(z, f as usize+z-1, 0);
                while l<=r { if l%2>0 {m=m.max(t[l]); l+=1} if r%2<1 {m=m.max(t[r]); r-=1} l/=2; r/=2 }
                Some((x-f).max(m) >= c[2])
            } else { let (l, r) = (*o.range(..x).next_back()?, *o.range(x..).next()?); o.insert(x);
                for (k,v) in [(l, x-l), (x, r-x)] { let mut i=k as usize+z; t[i]=v; while i>1 {i/=2; t[i]=t[i*2].max(t[i*2+1])} }
                None
            }
        }).collect()
    }
```

