---
layout: leetcode-entry
title: "1233. Remove Sub-Folders from the Filesystem"
permalink: "/leetcode/problem/2024-10-25-1233-remove-sub-folders-from-the-filesystem/"
leetcode_ui: true
entry_slug: "2024-10-25-1233-remove-sub-folders-from-the-filesystem"
---

[1233. Remove Sub-Folders from the Filesystem](https://leetcode.com/problems/remove-sub-folders-from-the-filesystem/description/) medium
[blog post](https://leetcode.com/problems/remove-sub-folders-from-the-filesystem/solutions/5965972/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/25102024-1233-remove-sub-folders?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-kPhrQHOPB8)
[deep-dive](https://notebooklm.google.com/notebook/20a5fc9f-38f6-4c68-a0df-22d0f94445d8/audio)
![1.webp](/assets/leetcode_daily_images/981365dc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/779

#### Problem TLDR

Remove empty subfolders #medium #trie #sort

#### Intuition

One way to do this in O(n) is to add everything into a Trie, mark the `ends`, then scan again and exclude path with more than one `end`.

Another way, is to sort paths, then naturally, every previous path will be parent of the next if it is a substring of it.

#### Approach

* Trie with keys of a `string` is faster in my tests then Trie with keys of individual `chars` (something with string optimizations)
* the fastest solution for this problem test cases is O(N(logN)), given the bigger constant of the Trie O(N) solution

#### Complexity

- Time complexity:
$$O(n)$$ for Trie, O(nlog(n)) for sort solution

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun removeSubfolders(folder: Array<String>) = buildList<String> {
        folder.sort()
        for (f in folder) if (size < 1 || !f.startsWith(last() + "/")) add(f)
    }

```
```rust

    pub fn remove_subfolders(mut folder: Vec<String>) -> Vec<String> {
        #[derive(Default)] struct Fs(u8, HashMap<String, Fs>);
        let (mut fs, mut res) = (Fs::default(), vec![]);
        for _ in 0..2 { for path in &folder {
            let mut r = &mut fs; let mut count = 0;
            for name in path.split('/').skip(1) {
                r = r.1.entry(name.into()).or_default();
                count += r.0
            }
            if r.0 == 1 && count == 1 { res.push(path.clone()) }
            r.0 = 1
        }}; res
    }

```
```c++

    vector<string> removeSubfolders(vector<string>& folder) {
        sort(begin(folder), end(folder)); vector<string> res;
        for (auto& f: folder)
            if (!size(res) || f.find(res.back() + "/"))
                res.push_back(f);
        return res;
    }

```

