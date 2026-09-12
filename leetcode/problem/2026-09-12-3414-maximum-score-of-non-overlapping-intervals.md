---
layout: leetcode-entry
title: "3414. Maximum Score of Non-overlapping Intervals"
permalink: "/leetcode/problem/2026-09-12-3414-maximum-score-of-non-overlapping-intervals/"
leetcode_ui: true
entry_slug: "2026-09-12-3414-maximum-score-of-non-overlapping-intervals"
---

[3414. Maximum Score of Non-overlapping Intervals](https://leetcode.com/problems/maximum-score-of-non-overlapping-intervals/solutions/8517154/kotlin-by-samoylenkodmitry-1oeg/) hard
[substack](https://dmitriisamoilenko.substack.com/p/12092026-3414-maximum-score-of-non?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/ULmIxJF3JE0)

https://dmitrysamoylenko.com/leetcode/

![12.09.2026.webp](/assets/leetcode_daily_images/12.09.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1480

#### Problem TLDR

Max lexi-small pick of at most 4 intervals

#### Intuition

DFS pick or skip, lookup for the next interval that is not intersecting the current. Compare sums and chosen indices.

#### Approach

* return pair of the sum and a list of indices

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun maximumWeight(iv: List<List<Int>>) = run {
        val ii = iv.indices.sortedBy{iv[it][0]}; val dp = HashMap<Int, Pair<Long,List<Int>>>()
        fun dfs(i: Int, c: Int): Pair<Long,List<Int>> = if(c<4&&i<iv.size)dp.getOrPut(i*4+c) {
            var j = ii.binarySearch{j -> if(iv[ii[i]][1]<iv[j][0])1 else -1}.inv()
            val skip = dfs(i+1,c); val (ts,tp) = dfs(j,c+1)
            val ns = ts+1L*iv[ii[i]][2]; val np = (tp+ii[i]).sorted(); val take = ns to np
            val c = skip.second.zip(np).find{(a,b)->a!=b}?.let{(a,b)->a<=b}?:(skip.second.size<=np.size)
            if (skip.first>ns) skip else if (skip.first<ns) take else if (c) skip else take
        } else 0L to listOf<Int>()
        dfs(0, 0).second
    }
```
```rust

```

