---
layout: leetcode-entry
title: "1233. Remove Sub-Folders from the Filesystem"
permalink: "/leetcode/problem/2025-07-19-1233-remove-sub-folders-from-the-filesystem/"
leetcode_ui: true
entry_slug: "2025-07-19-1233-remove-sub-folders-from-the-filesystem"
---

[1233. Remove Sub-Folders from the Filesystem](https://leetcode.com/problems/remove-sub-folders-from-the-filesystem/description) hard
[blog post](https://leetcode.com/problems/remove-sub-folders-from-the-filesystem/solutions/6977130/kotlin-rust-by-samoylenkodmitry-f106/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/19072025-1233-remove-sub-folders?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/MKiDuUcbb5M)
![1.webp](/assets/leetcode_daily_images/044687cc.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1054

#### Problem TLDR

Fold folders #medium

#### Intuition

Naive way: make set of folders, check each folder sub-paths to be in this set.
Clever way: sort folders, naturally the previous would be the parent if match.

#### Approach

* Trie gives a worse performance (HashMap based) 27ms vs 5ms

#### Complexity

- Time complexity:
$$O(nlogn)$$ or O(nl)

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 116ms
    fun removeSubfolders(f: Array<String>) = buildList<String> {
        val s = f.toSet()
        for (f in f) {
            var pref = ""; var skip = false
            for (x in f.split("/")) {
                pref += "$x"
                if (pref != f && pref in s) { skip = true; break }
                pref += "/"
            }
            if (!skip) add(f)
        }
    }

```
```kotlin

// 78ms
    fun removeSubfolders(f: Array<String>) = buildList<String> {
        f.sort()
        for (f in f) if (size < 1 || !f.startsWith("${last()}/")) this += f
    }

```
```rust

// 25ms
    pub fn remove_subfolders(mut f: Vec<String>) -> Vec<String> {
        #[derive(Default)] struct T(bool, HashMap<u8, T>);
        let (mut tr, mut r) = (T::default(), vec![]); f.sort_unstable();
        'o: for w in f { let mut t = &mut tr;
            for b in w.bytes().chain(once(b'/')) {
                t = t.1.entry(b).or_default(); if t.0 { continue 'o }
            }
            t.0 = true; r.push(w)
        } r
    }

```
```rust

// 5ms
    pub fn remove_subfolders(mut f: Vec<String>) -> Vec<String> {
        let mut r: Vec<String> = vec![]; f.sort_unstable();
        for f in f {
            if r.last().map_or(true, |l| f.len() < l.len() ||
                &f[..l.len()] != l || f.as_bytes()[l.len()] != b'/') { r.push(f) }
        } r
    }

```
```c++

// 59ms
    vector<string> removeSubfolders(vector<string>& f) {
        sort(begin(f), end(f)); vector<string> r;
        for (auto f: f) if (!size(r) || f.find(r.back() + "/")) r.push_back(f);
        return r;
    }

```

