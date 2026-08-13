---
layout: leetcode-entry
title: "2213. Longest Substring of One Repeating Character"
permalink: "/leetcode/problem/2026-08-13-2213-longest-substring-of-one-repeating-character/"
leetcode_ui: true
entry_slug: "2026-08-13-2213-longest-substring-of-one-repeating-character"
---

[2213. Longest Substring of One Repeating Character](https://leetcode.com/problems/longest-substring-of-one-repeating-character/solutions/8458201/kotlin-by-samoylenkodmitry-9a6q/) hard
[substack](https://dmitriisamoilenko.substack.com/p/13082026-2213-longest-substring-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rDeEHpkSBVk)

https://dmitrysamoylenko.com/leetcode/

![13.08.2026.webp](/assets/leetcode_daily_images/13.08.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1450

#### Problem TLDR

Queries of max repeating subarray after replacing characters

#### Intuition

Didn't solve.
```
    // 29 minute: i noticed we can disconnect groups
    //            union-find can't do that
    // from hint: segmented tree
```
Segment tree: in post-order traversal push up and merge left repeating suffix and right repeating prefix and max of left, right and suffix + prefix.
Merge suffixes and prefixes only if the entire subtree is repeating.

#### Approach

* c[mid] is the left last, c[mid+1] is the right first

#### Complexity

- Time complexity:
$$O(nlogn)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun longestRepeating(s: String, qC: String, qI: IntArray)=run {
        val c = s.toCharArray(); val mx = IntArray(4*c.size)
        val pr = IntArray(mx.size); val su = IntArray(mx.size)
        fun push(i:Int, l: Int, r: Int, mid: Int) {
            val ls = i*2; val rs = i*2+1; val match = c[mid]==c[mid+1]
            pr[i]=pr[ls]+if(match&& pr[ls]==mid-l+1) pr[rs] else 0
            su[i]=su[rs]+if(match&& su[rs]==r-mid) su[ls] else 0
            mx[i] = maxOf(mx[ls],mx[rs],if(match) su[ls]+pr[rs] else 0)
        }
        fun build(i: Int, l: Int, r: Int) {
            if (l==r) { mx[i]=1; pr[i]=1; su[i]=1; return}
            build(i*2,l,(l+r)/2);build(i*2+1,1+(l+r)/2,r); push(i,l,r,(l+r)/2)
        }
        fun update(i: Int, l: Int, r: Int, p: Int) {
            if (l==r) return; val mid = (l+r)/2
            if (p <= mid) update(i*2,l,mid,p) else update(i*2+1,mid+1,r,p)
            push(i,l,r,mid)
        }
        build(1, 0, c.size-1)
        qC.indices.map {c[qI[it]]=qC[it]; update(1,0,c.size-1,qI[it]); mx[1]}
    }
```
```rust

```

